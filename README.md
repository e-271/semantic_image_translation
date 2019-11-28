### Data set up
- Download the sentence embeddings from https://drive.google.com/open?id=1Bgnxqn0drNvIZgQ0-DjmzhP7mZAAVP7T 
```
unzip cub_sentence_embs.zip
mv cub_sentence_embs/train/*.csv ./datasets/edges2birds/
```


### Train
`python train.py --dataroot ./datasets/edges2birds --name test_pix2pix --model pix2pix --direction BtoA --dataset_mode semantic`
