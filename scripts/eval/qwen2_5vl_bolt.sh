#!/bin/bash

export DECORD_EOF_RETRY_MAX=20480

# videomme - frame 8
frame_indices_json=output/keyframes/videomme_clip_l_fps1_power3_frame8.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=8,frame_indices_json=$frame_indices_json \
    --tasks videomme \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f8

# videomme - frame 16
frame_indices_json=output/keyframes/videomme_clip_l_fps1_power3_frame16.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=16,frame_indices_json=$frame_indices_json \
    --tasks videomme \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f16

# videomme - frame 32
frame_indices_json=output/keyframes/videomme_clip_l_fps1_power3_frame32.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=32,frame_indices_json=$frame_indices_json \
    --tasks videomme \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f32


# longvideobench - frame 8
frame_indices_json=output/keyframes/longvideobench_clip_l_fps1_power2.5_frame8.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=8,frame_indices_json=$frame_indices_json \
    --tasks longvideobench_val_v \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f8

# longvideobench - frame 16
frame_indices_json=output/keyframes/longvideobench_clip_l_fps1_power2.5_frame16.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=16,frame_indices_json=$frame_indices_json \
    --tasks longvideobench_val_v \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f16

# longvideobench - frame 32
frame_indices_json=output/keyframes/longvideobench_clip_l_fps1_power2.5_frame32.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=32,frame_indices_json=$frame_indices_json \
    --tasks longvideobench_val_v \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f32


# mlvu - frame 8
frame_indices_json=output/keyframes/mlvu_clip_l_fps1_power3_frame8.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=8,frame_indices_json=$frame_indices_json \
    --tasks mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f8

# mlvu - frame 16
frame_indices_json=output/keyframes/mlvu_clip_l_fps1_power3_frame16.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=16,frame_indices_json=$frame_indices_json \
    --tasks mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f16

# mlvu - frame 32
frame_indices_json=output/keyframes/mlvu_clip_l_fps1_power3_frame32.json
accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,nframes=32,frame_indices_json=$frame_indices_json \
    --tasks mlvu_dev \
    --batch_size=1 \
    --log_samples \
    --output_path ./logs/qwen2_5vl_bolt_f32