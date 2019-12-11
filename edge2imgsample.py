from random import randint
import numpy as np
import torch
import argparse
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from infermodels import InferSent
from util.util import tensor2im, save_image
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, splitext
import glob

##############################################################
###################### Test options ########################
##############################################################

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
# opt.epoch = 130
# opt.dataroot = '/work/cascades/jiaruixu/dataset/edges2flowers'
## opt.dataroot = '/home/jiaruixu/git/InferSent/datasets/flower_samples'
### opt.checkpoints_dir = '/work/cascades/jiaruixu/pix2pix/'
# opt.sentence_file = 'flowersample.txt'
# opt.results_dir = './results'
## opt.results_dir = './results_sketch'
# opt.name = 'semanticv2_pix2pix_flowers'
# opt.model = 'semanticv2_pix2pix'
# opt.dataset_mode = 'semantic'

# Test birds
# opt.dataroot = '/work/cascades/jiaruixu/dataset/edges2birds'
# opt.sentence_file = 'birdsample.txt'
# opt.results_dir = './results_birds'
# opt.name = 'semanticv2_pix2pix_birds'

if not os.path.exists(opt.results_dir):
    os.mkdir(opt.results_dir)

##############################################################
###################### Embedding Genrator ########################
##############################################################

model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
# use_cuda = False
# model = model.cuda() if use_cuda else model
model = model.to(device)

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)
# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

# Load some sentences
sentences = []
with open(opt.sentence_file) as f:
    for line in f:
        sentences.append(line.strip())

embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))

##############################################################
###################### Image Genrator ########################
##############################################################
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

# test with eval mode. This only affects layers like batchnorm and dropout.
# For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
# For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
if opt.eval:
    model.eval()

input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

img_list=glob.glob(join(opt.dataroot, 'test', '*.jpg'))
# img = cv2.imread(join(img_path,'08111_AB.jpg'))
count = 1
total_img_num = len(img_list)

for AB_path in img_list:
    # AB_path = join(opt.dataroot, 'train', '5841_AB.jpg')
    print('processing image [%d/%d] from %s' % (count, total_img_num, AB_path))
    AB = Image.open(AB_path).convert('RGB')
    # split AB image into A and B
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    # A = AB
    # B = AB

    # apply the same transform to both A and B
    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))
    B_transform = get_transform(opt, transform_params, grayscale=(output_nc == 1))

    A = A_transform(A)
    B = B_transform(B)
    ch, w, h = A.size()

    data = {}
    data['A'] = A.type(torch.FloatTensor).view(-1, ch, w, h).to(device)
    data['B'] = B.type(torch.FloatTensor).view(-1, ch, w, h).to(device)
    data['A_paths'] = AB_path
    data['B_paths'] = AB_path

    for i in range(len(embeddings)):
        emb = embeddings[i].reshape(1, 4096)
        emb = 2 * (emb - np.min(emb)) / (np.max(emb) - np.min(emb)) - 1 # [-1, 1] scaling
        data['E'] = torch.FloatTensor(emb.reshape(1, 4096)).to(device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        model.set_input(data)
        model.test()           # run inference
        visuals = model.get_current_visuals()
        short_path = os.path.basename(AB_path)
        name = os.path.splitext(short_path)[0]
        for label, im_data in visuals.items():
            im = tensor2im(im_data)
            image_name = '%s_%s_%d.png' % (name, label, i)
            save_path = os.path.join(opt.results_dir, image_name)
            save_image(im, save_path, aspect_ratio=opt.aspect_ratio)
    print('saved results of %s to %s' % (name, opt.results_dir))
    # data['E'] = torch.FloatTensor(embeddings.mean(0).reshape(1, 4096)).to(device)
    # # self.image_paths = input['A_paths' if AtoB else 'B_paths']
    # i=0
    # model.set_input(data)
    # model.test()           # run inference
    # visuals = model.get_current_visuals()
    # short_path = os.path.basename(AB_path)
    # name = os.path.splitext(short_path)[0]
    # for label, im_data in visuals.items():
    #     im = tensor2im(im_data)
    #     image_name = '%s_%s_%d.png' % (name, label, i)
    #     save_path = os.path.join(opt.results_dir, image_name)
    #     save_image(im, save_path, aspect_ratio=opt.aspect_ratio)

    count += 1
    # if count == 2:
    #     break
