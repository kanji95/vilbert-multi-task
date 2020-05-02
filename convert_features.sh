#!/bin/bash
#SBATCH -A kanishk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=vigil
#SBATCH --mail-user=kanishk.jain@alumni.iiit.ac.in
#SBATCH --mail-type=ALL

module load cuda/10.0
module load cudnn/7-cuda-10.0

set -e

mkdir -p /ssd_scratch/cvit/kanishk/flickr30k_features
rm -rf /ssd_scratch/cvit/kanishk/flickr30k_features/*

echo "copying features from share3 to ssd_scratch"

scp -r kanishk@ada:/share3/kanishk/flickr30k_features /ssd_scratch/cvit/kanishk/

python3 script/convert_to_lmdb.py --features_dir /ssd_scratch/cvit/kanishk/flickr30k_features --lmdb_file /ssd_scratch/cvit/kanishk/flickr30k_lmdb

du -h /ssd_scratch/cvit/kanishk/flickr30k_lmdb

scp -r /ssd_scratch/cvit/kanishk/flickr30k_lmdb kanishk@ada:/share3/kanishk/

echo "copied features from ssd_scratch to share3"
