import os
import tqdm
import time
import argparse
import numpy as np
import datetime
import decord
import torch
from transformers import AutoModel, AutoTokenizer, SiglipProcessor, CLIPProcessor


class SigLip:
    def __init__(self, device="cuda"):
        self.model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            dtype=torch.float16,
            device_map=device,
        )
        self.processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        self.device = device
        self.model.eval()

    def extract_visual_features(self, images):
        # images: [bs,3,H,W]
        vision_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.autocast(self.device):
                image_features = self.model.get_image_features(**vision_inputs)
        return image_features  # [bs,C]

    def extract_text_features(self, text):
        # text: str
        text_inputs = self.tokenizer([text], truncation=True, padding="max_length", return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(self.device):
                text_features = self.model.get_text_features(**text_inputs)
        return text_features  # [C]

    def compute_similarity(self, video_features, text_features):
        # video_features: [T,C], text_features: [C]

        # normalized features
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_features, video_features.t().to(text_features.device))

        logit_scale = self.model.logit_scale.to(text_features.device)
        logit_bias = self.model.logit_bias.to(text_features.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        score = logits_per_text.detach().cpu().squeeze(0).float()  # [T]
        return score


class CLIP:
    def __init__(self, device="cuda"):
        self.model = AutoModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            dtype=torch.float16,
            device_map=device,
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device
        self.model.eval()

    def extract_visual_features(self, images):
        # images: [bs,3,H,W]
        vision_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.autocast(self.device):
                image_features = self.model.get_image_features(**vision_inputs)
        return image_features

    def extract_text_features(self, text):
        # text: str
        text_inputs = self.tokenizer([text], truncation=True, padding="max_length", return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(self.device):
                text_features = self.model.get_text_features(**text_inputs)
        return text_features

    def compute_similarity(self, video_features, text_features):
        # video_features: [T,C], text_features: [C]

        # normalized features
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_features, video_features.t().to(text_features.device))
        logits_per_text = logits_per_text * self.model.logit_scale.exp().to(text_features.device)
        score = logits_per_text.detach().cpu().squeeze(0).float()
        return score


def get_model(model_name, **kwargs):
    if model_name == "siglip":
        return SigLip(**kwargs)
    elif model_name == "clip":
        return CLIP(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Visual Feature")
    parser.add_argument("video_path", default="/data/", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--model", type=str, choices=["clip", "siglip"], default="clip")
    parser.add_argument("--fps", type=int, default=1, help="fps of the output video")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--part", type=int, default=0, help="all_data[part::total]")
    parser.add_argument("--total", type=int, default=1, help="all_data[part::total]")
    args = parser.parse_args()

    # print input and output
    print(f"Video path: {args.video_path}")
    print(f"Output path: {args.output_path}")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # build video list
    video_list = sorted(os.listdir(args.video_path))
    video_list = [x.replace(".mp4", "") for x in video_list if x.endswith(".mp4")]

    # partial video list
    video_list = video_list[args.part :: args.total]
    print("Processing part {} / {}".format(args.part, args.total))

    # check unprocessed video
    unprocessed_video = []
    for video_name in tqdm.tqdm(video_list):
        if os.path.exists(os.path.join(args.output_path, video_name + ".npy")):
            continue
        unprocessed_video.append(video_name)
    print(f"Unprocessed video number: {len(unprocessed_video)}")

    # build model
    device = "cuda"  # the device to load the model onto
    model = get_model(args.model, device=device)

    # do feature extraction
    print("\nFeature extraction starts....")
    start_time = time.time()
    for idx, video_name in enumerate(unprocessed_video):
        # read the video
        video_reader = decord.VideoReader(os.path.join(args.video_path, video_name + ".mp4"))
        video_fps = video_reader.get_avg_fps()
        total_frames = len(video_reader)

        # change fps
        new_total_frames = int(total_frames / video_fps * args.fps)
        frame_idxs = np.arange(0, new_total_frames, 1)
        frame_idxs = np.round(frame_idxs / args.fps * video_fps)  # new index that mapped with original fps
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).astype(int)

        # extract features
        feature_list = []
        for batch_start in range(0, len(frame_idxs), args.batch_size):
            # read frames
            batch_idxs = frame_idxs[batch_start : batch_start + args.batch_size]
            batch_images = video_reader.get_batch(batch_idxs).asnumpy()

            # extract visual features
            image_features = model.extract_visual_features(batch_images)
            feature_list.append(image_features.cpu().numpy())  # [T,C]

        video_features = np.concatenate(feature_list, axis=0)
        np.save(os.path.join(args.output_path, video_name + ".npy"), video_features)

        # print the eta time
        elapsed_time = time.time() - start_time
        eta_time = elapsed_time / (idx + 1) * (len(unprocessed_video) - idx)
        print(
            datetime.datetime.now(),
            "[{iter}/{max_iter}], time: {time}, eta: {eta}".format(
                time=str(datetime.timedelta(seconds=int(elapsed_time))),
                eta=str(datetime.timedelta(seconds=int(eta_time))),
                iter=idx + 1,
                max_iter=len(unprocessed_video),
            ),
        )
    print("\nFeature extraction finished....")
