#!/bin/bash
set -e
missing_id=$1

DATA_DIR="data/nuScenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="clr" # default settings

RGB_ENCODER="effb0"  # encoder for image(s)

python train_nuscenes_simplebev_imml.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=5000 \
       --dset='trainval' \
       --batch_size=1 \
       --grad_acc=5 \
       --data_dir=$DATA_DIR \
       --log_dir='logs_comparison' \
       --ckpt_dir='logs_comparison/ckpts' \
       --use_lidar=True \
       --use_radar=True \
       --use_metaradar=True \
       --res_scale=1 \
       --ncams=6 \
       --encoder_type=$RGB_ENCODER \
       --do_rgbcompress=True \
       --device_ids=[0] \
       --keep_latest=2 \
       --missing_id=$missing_id \

