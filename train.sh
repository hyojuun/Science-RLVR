#!/bin/bash

# 명령줄에서 학습 실행
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/ppo.py \
    --model_name_or_path Qwen/Qwen2.5-3B \
    --dataset_path /data/train.csv \
    --learning_rate 2.0e-05 \
    --num_train_epochs 1 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 16 \
    --mini_batch_size 8 \
    --batch_size 32 \
    --ppo_epochs 4 \
    --use_vllm True \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-3B-Open-R1-PPO

# YAML 설정 파일을 이용한 학습 실행
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/ppo.py \
    --config recipes/Qwen2.5-3B/ppo/config.yaml