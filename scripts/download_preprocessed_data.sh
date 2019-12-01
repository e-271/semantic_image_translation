#!/bin/bash

# Download training data
echo "Downloading edge2birds training data..."
echo "Downloading edge2birds sentence embeddings..."
python ./scripts/download_data.py 1SmxwLz11fUfHj_-Z8X7WYTwCgNjqjfc_ ./datasets/cub_sentence_embs.zip
echo "Downloading edge2flowers training data..."
python ./scripts/download_data.py 1omc4vK1mssnndHBGAYZ0FcnmAtcQf_PK ./datasets/flowers_samples.zip
echo "Downloading edge2flowers sentence embeddings..."
python ./scripts/download_data.py 1YAkfaGsue7hE-QGu0IqYTUAFbm_7MGRo ./datasets/flowers_semantic.zip

cd datasets
unzip cub_sentence_embs.zip
mkdir edges2birds
mv cub_sentence_embs/train/*.txt ./edges2birds/

unzip flowers_semantic.zip
mkdir edges2flowers
