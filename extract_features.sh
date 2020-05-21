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

echo "extracting features"

python3 script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir /home/kanishk/vigil/datasets/flickr30k_images --output_folder /ssd_scratch/cvit/kanishk/flickr30k_features

echo "copying features from ssd_scratch to share3"

scp -r /ssd_scratch/cvit/kanishk/flickr30k_features kanishk@ada:/share3/kanishk/
