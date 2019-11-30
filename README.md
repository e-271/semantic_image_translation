### Data set up
- CUB sentence embeddings: https://drive.google.com/file/d/1SmxwLz11fUfHj_-Z8X7WYTwCgNjqjfc_/view?usp=sharing
- Flowers sentence embeddings: https://drive.google.com/file/d/1YAkfaGsue7hE-QGu0IqYTUAFbm_7MGRo/view?usp=sharing

Also available in /work/cascades/erobb/flowers and /works/cascades/erobb/cub.

```
unzip cub_sentence_embs.zip
mv cub_sentence_embs/train/*.csv ./datasets/edges2birds/
```


### Train
`python train.py --dataroot ./datasets/edges2birds --name test_pix2pix --model pix2pix --direction BtoA --dataset_mode semantic`
