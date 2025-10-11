#!/bin/bash
set -e
modality=$1
missing_id=$2
sep_weight=$3
rec_weight=$4
sub_weight=$5

DATA_DIR="data/nuScenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="clr" # updated log dir

RGB_ENCODER="effb0"  # encoder for image(s)

python train_nuscenes_imml.py \
       --exp_name=${EXP_NAME} \
       --max_iters=50000 \
       --log_freq=50000 \
       --dset='trainval' \
       --batch_size=1 \
       --grad_acc=5 \
       --data_dir=$DATA_DIR \
       --log_dir='logs/' \
       --ckpt_dir='logs/ckpts' \
       --use_lidar=True \
       --use_radar=True \
       --use_metaradar=True \
       --res_scale=1 \
       --ncams=6 \
       --encoder_type=$RGB_ENCODER\
       --do_rgbcompress=True \
       --device_ids=[0] \
       --keep_latest=5 \
       --modality=$modality \
       --missing_id=$missing_id \
       --balanced_weights=[$sep_weight,$rec_weight,$sub_weight] \
       # --init_dir='logs/ckpts/1x5_3e-4s_20250826/ckpt_852/'

