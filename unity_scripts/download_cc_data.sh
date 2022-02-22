#!/bin/bash

#SBATCH --job-name=download_cc_data
#SBATCH --output=./mylogs/download_cc_data_stdout.txt
#SBATCH --partition=cpu
#SBATCH --mem=40G
#SBATCH --exclude=node92

set -e

cd /home/ahattimare_umass_edu/scratch/amit/SentAugment
mkdir data && cd data
source activate sent_augment
wget http://www.statmt.org/cc-english/x01.cc.5b.tar.gz

echo "end of downloading common crawl data script!"
