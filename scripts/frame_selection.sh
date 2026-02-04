# select keyframes for videomme
echo "Selecting 8 keyframes for VideoMME dataset..."
python -u select_frames.py \
    --model clip \
    --dataset videomme \
    --feat_folder output/features/clip_l_p14_fps1_videomme/ \
    --output output/keyframes/videomme_clip_l_fps1_power3_frame8.json \
    --fps 1 \
    --power 3 \
    --num_frames 8

echo -e "\nSelecting 16 keyframes for VideoMME dataset..."
python -u select_frames.py \
    --model clip \
    --dataset videomme \
    --feat_folder output/features/clip_l_p14_fps1_videomme/ \
    --output output/keyframes/videomme_clip_l_fps1_power3_frame16.json \
    --fps 1 \
    --power 3 \
    --num_frames 16

echo -e "\nSelecting 32 keyframes for VideoMME dataset..."
python -u select_frames.py \
    --model clip \
    --dataset videomme \
    --feat_folder output/features/clip_l_p14_fps1_videomme/ \
    --output output/keyframes/videomme_clip_l_fps1_power3_frame32.json \
    --fps 1 \
    --power 3 \
    --num_frames 32

# select keyframes for longvideobench
echo -e "\nSelecting 8 keyframes for LongVideoBench dataset..."
python -u select_frames.py \
    --model clip \
    --dataset longvideobench \
    --feat_folder output/features/clip_l_p14_fps1_longvideobench/ \
    --output output/keyframes/longvideobench_clip_l_fps1_power2.5_frame8.json \
    --fps 1 \
    --power 2.5 \
    --num_frames 8

echo -e "\nSelecting 16 keyframes for LongVideoBench dataset..."
python -u select_frames.py \
    --model clip \
    --dataset longvideobench \
    --feat_folder output/features/clip_l_p14_fps1_longvideobench/ \
    --output output/keyframes/longvideobench_clip_l_fps1_power2.5_frame16.json \
    --fps 1 \
    --power 2.5 \
    --num_frames 16

echo -e "\nSelecting 32 keyframes for LongVideoBench dataset..."
python -u select_frames.py \
    --model clip \
    --dataset longvideobench \
    --feat_folder output/features/clip_l_p14_fps1_longvideobench/ \
    --output output/keyframes/longvideobench_clip_l_fps1_power2.5_frame32.json \
    --fps 1 \
    --power 2.5 \
    --num_frames 32

# select keyframes for mlvu
echo -e "\nSelecting 8 keyframes for MLVU dataset..."
python -u select_frames.py \
    --model clip \
    --dataset mlvu \
    --feat_folder output/features/clip_l_p14_fps1_mlvu/ \
    --output output/keyframes/mlvu_clip_l_fps1_power3_frame8.json \
    --fps 1 \
    --power 3 \
    --num_frames 8

echo -e "\nSelecting 16 keyframes for MLVU dataset..."
python -u select_frames.py \
    --model clip \
    --dataset mlvu \
    --feat_folder output/features/clip_l_p14_fps1_mlvu/ \
    --output output/keyframes/mlvu_clip_l_fps1_power3_frame16.json \
    --fps 1 \
    --power 3 \
    --num_frames 16

echo -e "\nSelecting 32 keyframes for MLVU dataset..."
python -u select_frames.py \
    --model clip \
    --dataset mlvu \
    --feat_folder output/features/clip_l_p14_fps1_mlvu/ \
    --output output/keyframes/mlvu_clip_l_fps1_power3_frame32.json \
    --fps 1 \
    --power 3 \
    --num_frames 32