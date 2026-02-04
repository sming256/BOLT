#!/bin/bash

export DECORD_EOF_RETRY_MAX=20480

# frame 8
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,nframes=8 \
    --tasks videomme,longvideobench_val_v,mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen3vl_baseline_f8

# frame 16
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,nframes=16 \
    --tasks videomme,longvideobench_val_v,mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen3vl_baseline_f16

# frame 32
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,nframes=32 \
    --tasks videomme,longvideobench_val_v,mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen3vl_baseline_f32

