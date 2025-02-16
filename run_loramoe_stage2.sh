lr=2e-4
lora_rank=4
lora_alpha=8
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.05
lora_nums=5
blc_alpha=0.0
blc_weight=0.0


pretrained_model=
tokenizer_path=
dataset_dir=
exp_name=
setting=
output_dir=
max_seq_length=

per_device_train_batch_size=1

per_device_eval_batch_size=1
gradient_accumulation_steps=1
deepspeed_config_file=ds_zero2_no_offload.json
echo ${output_dir}/${exp_name}/${setting}
mkdir -p ${output_dir}/${exp_name}/${setting}
mkdir -p output/log/${exp_name}

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 --master_port 29503 \
    run_loramoe.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed 41 \
    --bf16 \
    --num_train_epochs 2 \
    --report_to none \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 20 \
    --save_steps 2000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name}${setting} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_nums} \
    --blc_alpha ${blc_alpha} \
    --blc_weight ${blc_weight} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype bfloat16 \
    --setting ${setting} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --overwrite_output_dir \
    2>&1 | tee output/log/${exp_name}/${setting}.log