### Installation
For conda users, you can create an environment using
```
conda create -n pix2pix python=3.5.5
bash ./scripts/conda_deps.sh
```

### Download Dataset

```
bash ./scripts/download_preprocessed_data.sh
```

### Train
Train with semantic pix2pix
```
python train.py \
      --dataroot /work/cascades/jiaruixu/dataset/edges2flowers \
      --name semantic_pix2pix \
      --model semantic_pix2pix \
      --direction AtoB \
      --dataset_mode semantic \
      --checkpoints_dir /work/cascades/jiaruixu/pix2pix/ \
      --display_id -1
```

Train with pix2pix
```
python train.py \
      --dataroot /work/cascades/jiaruixu/dataset/edges2flowers \
      --name pix2pix \
      --model pix2pix \
      --direction AtoB \
      --dataset_mode semantic \
      --checkpoints_dir /work/cascades/jiaruixu/pix2pix/ \
      --display_id -1
```

### Tensorboard for training logs

```
tensorboard --logdir=/work/cascades/jiaruixu/pix2pix/[exp_name]/logs
```

### Test

```
python test.py \
       --dataroot /work/cascades/jiaruixu/dataset/edges2flowers \
       --name semantic_pix2pix \
       --model semantic_pix2pix \
       --dataset_mode semantic \
       --direction AtoB
```

### Acknowledgments
Our code is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
