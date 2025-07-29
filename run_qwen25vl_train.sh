export RUN_NAME="VersaVid-R1_final"
export BASE_MODEL_NAME="path_to/Qwen2.5-VL-7B-Instruct"
export DATA_PATH="path_to_jsonl"
ts=`date +%Y_%m_%d_%H_%M`
export OUT_DIR="path_to/VersaVid-R1/ckpt/${BASE_MODEL_NAME}/${RUN_NAME}_${ts}"

mkdir -p $OUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12353" \
    src/open_r1_video/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUT_DIR \
    --model_name_or_path $BASE_MODEL_NAME \
    --dataset_name xxx \
    --jsonl_path $DATA_PATH \
    --max_prompt_length 8192 \
    --max_completion_length 2048 \
    --learning_rate 1e-6 \
    --beta 0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 8 \
    --report_to tensorboard 2>&1 | tee path_to/VersaVid-R1/train_log/${BASE_MODEL_NAME}/exp_log_${RUN_NAME}_${ts}.log \