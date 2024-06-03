MODEL=mistralai/Mistral-7B-Instruct-v0.2
train_file=../datasets/stage_1/sft_reason_conf.jsonl
NUM_GPUS=4
BATCH_SIZE_PER_GPU=8
LEARNING_RATE=7e-5
TOTAL_BATCH_SIZE=128 # max 2 for 2 GPUs
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
OUTPUT_DIR=model_lr${LEARNING_RATE}_bs${TOTAL_BATCH_SIZE}
echo "Training ${MODEL} model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ../ds_configs/stage3_no_offloading_accelerate.conf \
    ./finetune.py \
    --model_name_or_path $MODEL \
    --tokenizer_name $MODEL \
    --use_slow_tokenizer \
    --train_file ${train_file} \
    --max_seq_length 400 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --output_dir output/${OUTPUT_DIR}/ \
