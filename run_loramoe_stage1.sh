export CUDA_VISIBLE_DEVICES=0
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO

model_name_or_path=
dataset_name=
output_dir=
max_seq_length=

accelerate launch  --config_file config.yaml\
    --main_process_port 29501 \
    run_lora_tuning.py \
    --model_name_or_path=${model_name_or_path} \
    --torch_dtype="bfloat16" \
    --dataset_name=${dataset_name} \
    --report_to="none" \
    --learning_rate=3e-4 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --output_dir=${output_dir} \
    --logging_steps=20 \
    --num_train_epochs=3 \
    --weight_decay=0.1 \
    --warmup_ratio=0.03 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --max_seq_length=${max_seq_length} \
    --bf16  \
    --use_peft \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_target_modules q_proj k_proj v_proj o_proj