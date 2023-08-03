import argparse

from train_net_semi import train_semi,test

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training and testing pipeline."
    )
    
    # setting
    parser.add_argument(
        '--training', 
        action='store_true', 
        help='Training enable.'
    )
    parser.add_argument(
        '--local_rank', 
        type=int, 
        help='Local rank for ddp.'
    )
    parser.add_argument(
        '--backend', 
        default='nccl', 
        type=str, 
        help='Backend for ddp.'
    )
    parser.add_argument(
        '--seed', 
        default=3407, 
        type=int, 
        help='Random Seed.'
    )
    parser.add_argument(
        '--num_gpus',
        default=4, 
        type=int,
        help='Number of GPUs to use (applies to both training and testing).'
    )

    # model
    parser.add_argument(
        '--detectron2_ckpt', 
        default='./pretrained_models/fpn/model_final_cafdb1.pkl', 
        type=str, 
        help='ckpt path of fpn from detectron2.'
    )
    parser.add_argument(
        '--detectron2_cfg', 
        default='./configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_train.yaml',
        type=str, 
        help='cfg path of fpn from detectron2.'
    )
    parser.add_argument(
        '--max_sequence_length',
        default=230,
        type=int,
        help='Max length of the input language sequence.'
    )
    parser.add_argument(
        '--max_seg_num',
        default=64,
        type=int,
        help='Max num of the noun phrase to be segmented.'
    )
    parser.add_argument(
        '--max_phrase_num',
        default=30,
        type=int,
        help='Max num of the noun phrase to be segmented.'
    )
    parser.add_argument(
        '--pretrained_bert',
        default='./pretrained_models/bert/bert-base-uncased',
        type=str,
        help='Pretrained bert model.'
    )
    parser.add_argument(
        '--bert_tokenize',
        default='./pretrained_models/bert/bert-base-uncased.txt',
        type=str,
        help='Tokenize word list.'
    )
    parser.add_argument(
        '--bert_freeze',
        default = False,
        action='store_true',
        help='If true freeze BERT model.'
    )
    parser.add_argument(
        '--fpn_freeze',
        default = True,
        action='store_true',
        help='If true freeze FPN model.'
    )
    
    parser.add_argument(
        '--ckpt_path',
        default='',
        type=str,
        help='Path to the checkpoint to load the initial weight.'
    )
    parser.add_argument(
        '--num_stages',
        default=3,
        type=int,
        help='Iter num.'
    )
    parser.add_argument(
        '--num_points',
        default=200,
        type=int,
        help='Saliency Points num.'
    )

    # data
    parser.add_argument(
        '--data_path',
        default='./datasets/coco',
        type=str,
        help='The path to the data directory.'
    )
    
    parser.add_argument(
        '--data_dir',
        default='./datasets',
        type=str,
        help='The path to the data directory.'
    )
    parser.add_argument(
        '--batch_size',
        default=12, 
        type=int,
        help='Total mini-batch size.'
    )
    parser.add_argument(
        '--num_workers',
        default=0, 
        type=int,
        help='Number of data loader workers per training process.'
    )
    parser.add_argument(
        '--pin_memory', 
        default=True,
        type=bool,
        help='Load data to pinned host memory.'
    )

    # training pipeline
    parser.add_argument(
        '--epoch',
        default=14,
        type=int,
        help='Training epoch.'
    )
    parser.add_argument(
        '--base_lr',
        default=1e-4, 
        type=float,
        help='Learning rate.'
    )
    parser.add_argument(
        '--weight_decay',
        default=0, 
        type=float,
        help='Weight decay.'
    )
    parser.add_argument(
        '--scheduler',
        default='step', 
        choices=['step', 'reduce'],
        type=str,
        help='Weight decay.'
    )

    # output
    parser.add_argument(
        '--save_fig',
        action='store_true',
        help='Saving evaluation figures of metrics.'
    )
    parser.add_argument(
        '--save_ckpt',
        default=9,
        type=int,
        help='Epoch for starting saving checkpoints.'
    )
    parser.add_argument(
        '--log_period',
        default=100,
        type=int,
        help='Logging period.'
    )
    parser.add_argument( 
        '--output_dir', 
        default="./output", 
        type=str, 
        help='Saving dir.'
    )

    # semi-supervised learning settings
    parser.add_argument(
        '--semi_cfg',
        type=str,
        help='config for semi supervised learning'
    ) 


    return parser.parse_args()

def main():
    args = parse_args()
    if args.training:       # args.training = True
        train_semi(args)
    else:                   # args.training = False
        test(args)      

if __name__ == "__main__":
    main()