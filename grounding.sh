#!/bin/bash
#SBATCH -A kanishk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --job-name=vigil
#SBATCH --mail-user=kanishk.jain@alumni.iiit.ac.in
#SBATCH --mail-type=ALL

module load cuda/10.0
module load cudnn/7-cuda-10.0

set -e

if [ -d "/ssd_scratch/cvit/kanishk/flickr30k" ]; then
    echo "folder exists, proceed with training"
else
    mkdir -p /ssd_scratch/cvit/kanishk/
    rm -rf /ssd_scratch/cvit/kanishk/*

    echo "copying features from share3 to ssd_scratch"
    scp -r kanishk@ada:/share3/kanishk/flickr30k /ssd_scratch/cvit/kanishk/
    echo "copied features from scratch3"
fi

#python3 train_tasks.py --bert_model bert-base-uncased --from_pretrained data/multi_task_model.bin --config_file config/bert_base_6layer_6conect.json --resume_file save/FlickrGrounding_bert_base_6layer_6conect-finetune_from_multi_task_model/pytorch_ckpt_latest.tar --tasks 18 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model

#python3 eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/FlickrGrounding_bert_base_6layer_6conect-finetune_from_multi_task_model/pytorch_model_13.bin --config_file config/bert_base_6layer_6conect.json --resume_file save/FlickrGrounding_bert_base_6layer_6conect-finetune_from_multi_task_model/pytorch_ckpt_latest.tar --tasks 18 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model

python3 eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/FlickrGrounding_bert_base_6layer_6conect-finetune_from_multi_task_model/pytorch_model_13.bin --config_file config/bert_base_6layer_6conect.json --tasks 18
