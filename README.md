# BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding

<p align="left">
<a href="https://arxiv.org/abs/2503.21483" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.21483-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/datasets/sming256/BOLT_data" alt="data">
    <img src="https://img.shields.io/badge/Data-HuggingFace-yellow.svg" /></a>
</p>

This is the official implementation of the paper [BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding](https://arxiv.org/abs/2503.21483), which is accepted by CVPR2025.

## Abstract

Large video-language models (VLMs) have demonstrated promising progress in various video understanding tasks. However, their effectiveness in long-form video analysis is constrained by limited context windows. Traditional approaches, such as uniform frame sampling, often inevitably allocate resources to irrelevant content, diminishing their effectiveness in real-world scenarios. In this paper, we introduce BOLT, a method to BOost Large VLMs without additional Training through a comprehensive study of frame selection strategies. 

First, to enable a more realistic evaluation of VLMs in long-form video understanding, we propose a multi-source retrieval evaluation setting. Our findings reveal that uniform sampling performs poorly in noisy contexts, underscoring the importance of selecting the right frames. Second, we explore several frame selection strategies based on query-frame similarity and analyze their effectiveness at inference time. Our results show that inverse transform sampling yields the most significant performance improvement, increasing accuracy on the Video-MME benchmark from 53.8% to 56.1% and MLVU benchmark from 58.9% to 63.4%.

<p align="center">
    <img src="assets/method.jpg" width="80%"></a> <br>
</p>

## Installation

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) as our evaluation framework. Follow these steps to set up the environment:

```bash
# Create and activate conda environment
conda create -n bolt python=3.12
conda activate bolt

# Clone the repository with submodules
git clone --recurse-submodules https://github.com/your-username/BOLT.git
cd BOLT

# Install dependencies
pip install -e third_party/lmms-eval
pip install -e third_party/LLaVA-NeXT/
pip install -e third_party/qwen-vl-utils/
```
## Implementation Details

The inverse transform sampling in BOLT is implemented in [here](./select_frames.py#L76-L94). 

The frame selection for LLaVA-OneVision/Qwen2.5-VL/Qwen3-VL under lmms-eval framework is implemented in [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/compare/main...sming256:lmms-eval:main).


## Evaluation

### 1. Prepare Video Features for Frame Selection

We use the CLIP ViT-L/14 image encoder to extract frame features for frame selection. Before running the script, please download the raw video datasets and place them in the corresponding paths as specified in the bash file.

```bash
bash scripts/feature_extraction.sh
```

**Note:**  You can also directly download our pre-extracted video features from [HuggingFace](https://huggingface.co/datasets/sming256/BOLT_data).

### 2. Frame Selection with BOLT

By using inverse transform sampling based on query-frame similarity, we select the most informative frames for each video.
```bash
bash scripts/frame_selection.sh
```

**Note:** You can also directly download our pre-computed keyframe indices from [HuggingFace](https://huggingface.co/datasets/sming256/BOLT_data).

### 3. Model Evaluation

Baseline Evaluation (Uniform Sampling)
```bash
# LLaVA-OneVision with uniform sampling (8/16/32 frames)
bash scripts/eval/llava_ov_baseline.sh

# Qwen2.5-VL with uniform sampling (8/16/32 frames)
bash scripts/eval/qwen2_5_vl_baseline.sh

# Qwen3-VL with uniform sampling (8/16/32 frames)
bash scripts/eval/qwen3_vl_baseline.sh
```

BOLT Evaluation (Our Method)
```bash
# LLaVA-OneVision with BOLT (8/16/32 frames)
bash scripts/eval/llava_ov_bolt.sh

# Qwen2.5-VL with BOLT (8/16/32 frames)
bash scripts/eval/qwen2_5_vl_bolt.sh

# Qwen3-VL with BOLT (8/16/32 frames)
bash scripts/eval/qwen3_vl_bolt.sh
```

## Performance Results

Video-MME Benchmark
| Model                  | Sampling Method | Acc (8 frames)  | Acc (16 frames) | Acc (32 frames) |
| ---------------------- | --------------- | --------------- | --------------- | --------------- |
| LLaVA-OneVision-7B     | Uniform         | 54.0            | 56.7            | 58.5            |
| **LLaVA-OneVision-7B** | **BOLT**        | **56.1 (+2.1)** | **58.3 (+1.6)** | **59.5 (+1.0)** |
| Qwen2.5-VL-7B          | Uniform         | 53.8            | 58.8            | 62.2            |
| **Qwen2.5-VL-7B**      | **BOLT**        | **57.4 (+3.6)** | **60.9 (+2.1)** | **64.0 (+1.8)** |
| Qwen3-VL-8B            | Uniform         | 56.0            | 60.5            | 64.3            |
| **Qwen3-VL-8B**        | **BOLT**        | **58.9 (+2.9)** | **62.7 (+2.2)** | **65.7 (+1.4)** |

LongVideoBench Benchmark
| Model                  | Sampling Method | Acc (8 frames)  | Acc (16 frames) | Acc (32 frames) |
| ---------------------- | --------------- | --------------- | --------------- | --------------- |
| LLaVA-OneVision-7B     | Uniform         | 54.2            | 56.0            | 56.6            |
| **LLaVA-OneVision-7B** | **BOLT**        | **54.5 (+0.3)** | **56.7 (+0.7)** | **58.1 (+1.5)** |
| Qwen2.5-VL-7B          | Uniform         | 53.2            | 56.1            | 58.6            |
| **Qwen2.5-VL-7B**      | **BOLT**        | **55.1 (+1.9)** | **57.5 (+1.4)** | **60.0 (+1.4)** |
| Qwen3-VL-8B            | Uniform         | 54.8            | 57.7            | 60.4            |
| **Qwen3-VL-8B**        | **BOLT**        | **57.4 (+2.6)** | **58.9 (+1.2)** | **62.2 (+1.8)** |

MLVU Benchmark
| Model                  | Sampling Method | Acc (8 frames)  | Acc (16 frames) | Acc (32 frames) |
| ---------------------- | --------------- | --------------- | --------------- | --------------- |
| LLaVA-OneVision-7B     | Uniform         | 58.4            | 60.9            | 63.1            |
| **LLaVA-OneVision-7B** | **BOLT**        | **63.6 (+5.2)** | **66.1 (+5.2)** | **66.8 (+3.7)** |
| Qwen2.5-VL-7B          | Uniform         | 57.1            | 59.9            | 62.5            |
| **Qwen2.5-VL-7B**      | **BOLT**        | **63.1 (+6.0)** | **66.2 (+6.3)** | **68.4 (+5.9)** |
| Qwen3-VL-8B            | Uniform         | 55.3            | 58.9            | 63.8            |
| **Qwen3-VL-8B**        | **BOLT**        | **62.9 (+7.6)** | **67.7 (+8.8)** | **70.3 (+6.5)** |

## Inference Demo

We provide a simple demo to showcase the inference process of BOLT, including the visualization of selected frames. You can run the demo with the following command:

```
# download demo video
hf download --repo-type dataset MLVU/MVLU \
    --include "MLVU/video/1_plotQA/movie101_58.mp4" \
    --local-dir assets

# run demo
python demo.py \
    --video_path assets/MLVU/video/1_plotQA/movie101_58.mp4 \
    --query "At the end of the video, what happens to the van?"
```
Please refer to [demo.py](demo.py) for more details.

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{liu2025bolt,
    title     = {BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding},
    author    = {Liu, Shuming and Zhao, Chen and Xu, Tianqi and Ghanem, Bernard},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```

## Contact

If you have any questions or suggestions, please feel free to contact us at: `shuming.liu@kaust.edu.sa`.
