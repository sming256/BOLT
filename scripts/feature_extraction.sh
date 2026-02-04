# extract CLIP visual features for videomme
python -u extract_feature.py \
    $HF_HOME/videomme/data/ \
    output/features/clip_l_p14_fps1_videomme \
    --model clip \
    --fps 1

# extract CLIP visual features for longvideobench
python -u extract_feature.py \
    $HF_HOME/longvideobench/videos/ \
    output/features/clip_l_p14_fps1_longvideobench \
    --model clip \
    --fps 1

# extract CLIP visual features for mlvu
python -u extract_feature.py \
    $HF_HOME/mlvu/ \
    output/features/clip_l_p14_fps1_mlvu \
    --model clip \
    --fps 1