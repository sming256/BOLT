#!/bin/bash

export DECORD_EOF_RETRY_MAX=20480

# frame 8
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,max_frames_num=8 \
    --tasks videomme,longvideobench_val_v,mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/llava_ov_baseline_f8

# frame 16
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,max_frames_num=16 \
    --tasks videomme,longvideobench_val_v,mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/llava_ov_baseline_f16

# frame 32
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,max_frames_num=32 \
    --tasks videomme,longvideobench_val_v,mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/llava_ov_baseline_f32

