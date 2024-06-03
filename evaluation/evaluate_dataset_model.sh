MODEL=$1
DATASET_FILE=$2
NUM_GPUS=$5
BATCH_SIZE_PER_GPU=$6
OUTPUT_DIR=$7
echo "Evaluating model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --main_process_port 29501 \
    ./evaluate.py \
    --model_name_or_path $MODEL \
    --tokenizer_name mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_name $DATASET_FILE \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --train_micro_batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to wandb \
    --num_eval_examples 1000 \
