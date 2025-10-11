set -e
# run_idx=$1
# gpu=$2

run_idx=1
gpu=0

for i in `seq 1 1 9`;
do

cmd="python3 train_miss_mce.py 
--gpu_ids=$gpu --has_test --name=mce_IEMOCAP 
--checkpoints_dir=logs/checkpoints/$i --log_dir=logs/$i

--model=mce --weight_decay=1e-5
--dataset_mode=multimodal_miss --num_thread=0 --batch_size=128 --in_mem

--verbose --suffix=block_{n_blocks}_run{run_idx}_mce
--print_freq=10 --niter=30 --niter_decay=30 --lr=2e-4

--run_idx=$run_idx --rate_indx=$i
--record_folder=vis/mce_IEMOCAP$i

--cvNo=1
--A_type=comparE --V_type=denseface --L_type=bert_large 
--norm_method=trn --output_dim=4 
--corpus_name=IEMOCAP 

--input_dim_a=130 --input_dim_l=1024 --input_dim_v=342 
--embd_size_a=128 --embd_size_l=128 --embd_size_v=128 
--embd_method_a=maxpool --embd_method_v=maxpool
--AE_layers=256,128,64 --cls_layers=128,128 --n_blocks=5 --dropout_rate=0.5
--pretrained_path='checkpoints/CAP_utt_fusion_AVL_run1'
--ce_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0

--rec_weight=1 --sep_weight=1 --sub_weight=2
--use_a --use_b

--select_seed=$i
"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done