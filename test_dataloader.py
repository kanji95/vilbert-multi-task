import argparse
import json
import logging
import os
import random
from io import open
from bisect import bisect

from time import gmtime, strftime
from timeit import default_timer as timer

from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer

from torch.nn import CrossEntropyLoss

from vilbert.vilbert import BertConfig
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
from vilbert.datasets.flickr_grounding_dataset import FlickrGroundingDataset

from block import fusions

from vilbert.vilbert import BertModel, BertPreTrainedModel

from pprint import pprint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Vilbert(BertPreTrainedModel):
    def __init__(self, config, max_num_region=200, chunks=50, default_gpu=True, dropout_prob=0.1):
        super(Vilbert, self).__init__(config)

        self.vilbert = BertModel(config)
        self.fusion = fusions.Block(input_dims=[max_num_region*config.v_hidden_size, config.bi_hidden_size],
                                    output_dim=max_num_region,
                                    mm_dim=512,
                                    chunks=100)
        self.dropout = nn.Dropout(dropout_prob)
        self.vision_logit = nn.Linear(config.v_hidden_size, 1)

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        task_ids=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):

        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.vilbert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            co_attention_mask,
            task_ids,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        # print(sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask)
        print(f'sequence_output_t: {sequence_output_t.shape}, sequence_output_v: {sequence_output_v.shape}, pooled_output_t: {pooled_output_t.shape}, pooled_output_v: {pooled_output_v.shape}')
        
        vision_features = sequence_output_v.view(-1, 200*1024)
        text_features = pooled_output_t
        input_features = [vision_features, text_features]

        fused_features = self.fusion(input_features).unsqueeze(dim=2)
        
        vision_logit = self.vision_logit(self.dropout(sequence_output_v)) + (
            (1.0 - image_attention_mask) * -10000.0
        ).unsqueeze(2).to(dtype=next(self.parameters()).dtype)

        return sequence_output_v, pooled_output_t, fused_features, vision_logit

def main():
    parser = argparse.ArgumentParser()

    # Data files for FOIL task.
    parser.add_argument(
        "--features_h5path",
        default="data/datasets/flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb",
    )
    parser.add_argument(
        "--gt_features_h5path",
        default="data/datasets/flickr30k/flickr30k_gt_resnext152_faster_rcnn_genome.lmdb",
    )

    parser.add_argument("--instances-jsonpath", default="data/referExpression")
    parser.add_argument("--task", default="refcoco+")

    # Required parameters
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--pretrained_weight",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=30,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=128,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--from_pretrained",
        action="store_true",
        help="Wheter the tensor is from pretrained.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Wheter to use the baseline model (single bert).",
    )

    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )

    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="whether use chunck for parallel training.",
    )

    args = parser.parse_args()

    # Declare path to save checkpoints.
    config = BertConfig.from_json_file(args.config_file)
    # print(config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    num_train_optimization_steps = None

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    image_features_reader = ImageFeaturesH5Reader(args.features_h5path, True)
    gt_image_features_reader = ImageFeaturesH5Reader(
        args.gt_features_h5path, True)

    dataset = FlickrGroundingDataset(task="FlickrGrounding",
                                     dataroot="data/datasets/flickr30k/",
                                     annotations_jsonpath="",
                                     split="val",
                                     image_features_reader=image_features_reader,
                                     gt_image_features_reader=gt_image_features_reader,
                                     tokenizer=tokenizer,
                                     bert_model=args.bert_model,
                                     clean_datasets=True,
                                     padding_index=0,
                                     max_seq_length=24,
                                     max_region_num=200,
                                     )

    dataloader = DataLoader(dataset=dataset, batch_size=4,
                            shuffle=False, pin_memory=True)

    dataset_iter = iter(dataloader)

    features, spatials, image_mask, caption, target, input_mask, segment_ids, co_attention_mask, image_id, grid_vec = next(dataset_iter)
    
    pprint(f'features: {features.shape}, spatials: {spatials.shape}, image_mask: {image_mask.shape}, caption: {caption.shape}, target: {target.shape}, input_mask: {input_mask.shape}, segment_ids: {segment_ids.shape}, co_attention_mask: {co_attention_mask.shape}, image_id: {image_id.shape}, grid_vec: {grid_vec.shape}')
    
    
    ### Model Description ###
    config.v_target_size = 1601
    config.visual_target = 0
    config.task_specific_tokens = True
    config.dynamic_attention = True
    config.model = "bert"
    
    task_tokens = caption.new().resize_(caption.size(0), 1).fill_(18)
    
    #model = Vilbert(config=config, default_gpu=False)
    
    #model(caption, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens)
    
if __name__ == "__main__":

    main()
