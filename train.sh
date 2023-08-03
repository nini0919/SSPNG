# config setting
EXP_NAME=exp_demo
# EXP_NAME=debug

PROJECT_ROOT=.
DETECTRON2_CKPT=${PROJECT_ROOT}/pretrained_models/fpn/model_final_cafdb1.pkl
OUTPUT_DIR=${PROJECT_ROOT}/${EXP_NAME}
BERT=${PROJECT_ROOT}/pretrained_models/bert/bert-base-uncased
BERT_TOKENIZE=${PROJECT_ROOT}/pretrained_models/bert/bert-base-uncased.txt
DATA_PATH=${PROJECT_ROOT}/datasets/coco
SEMI_CONFIG=${PROJECT_ROOT}/${EXP_NAME}/config.yaml 

# semi settings config 
NODES=1
NUM_GPUS=4
BATCH_SIZE=12

# traiging
 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --nnodes=${NODES}  \
     main_semi.py --training \
     --num_gpus ${NUM_GPUS} \
     --batch_size ${BATCH_SIZE} \
     --fpn_freeze \
     --output_dir ${OUTPUT_DIR} \
     --detectron2_ckpt ${DETECTRON2_CKPT} \
     --pretrained_bert ${BERT} \
     --bert_tokenize ${BERT_TOKENIZE} \
     --data_path ${DATA_PATH} \
     --num_points 200 \
     --num_stages 3 \
     --semi_cfg ${SEMI_CONFIG}\
     $@