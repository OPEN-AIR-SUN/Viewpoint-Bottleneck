#!/bin/bash

set -exo pipefail

PRETRAIN_PATH=${1:-'/data/luoly/dataset/final/min_weights/bt/512/weights.pth'}
DATASET_PATH=${2:-'/data/luoly/dataset/Min_scan'}
LOG_DIR=${3:-'logs'}
SHOTS=${4:-200}
BATCH_SIZE=${5:-8}
MODEL=${6:-'Res16UNet34C'}

# TODO specify 
DATAPATH=$DATASET_PATH/eff_$SHOTS/train
TESTDATAPATH=$DATASET_PATH/scan_processed/train
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'4'}


# TODO remove distributed_world_size argument
python -m ddp_main \
    --distributed_world_size=1 \
    --train_phase=train \
    --is_train=True \
    --lenient_weight_loading=True \
    --stat_freq=1 \
    --val_freq=500 \
    --save_freq=500 \
    --model=${MODEL} \
    --conv1_kernel_size=5 \
    --normalize_color=True \
    --dataset=ScannetVoxelization2cmDataset \
    --testdataset=ScannetVoxelization2cmtestDataset \
    --batch_size=$BATCH_SIZE \
    --num_workers=1 \
    --num_val_workers=1 \
    --scannet_path=${DATAPATH} \
    --scannet_test_path=${TESTDATAPATH} \
    --return_transformation=False \
    --test_original_pointcloud=False \
    --save_prediction=False \
    --lr=0.1 \
    --scheduler=PolyLR \
    --max_iter=30000 \
    --log_dir=${LOG_DIR} \
    --weights=${PRETRAIN_PATH} \
     2>&1 | tee -a "$LOG"
