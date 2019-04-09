#! /bin/sh
# copied from https://github.com/1Konny/Beta-VAE/blob/master/download_dsprites.sh

mkdir data
cd data
git clone https://github.com/deepmind/dsprites-dataset.git
cd dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
