import argparse
import os
import torch
import numpy as np
import tqdm
import pandas as pd
import json
from extract_feature import get_model


def get_query_list(dataset, video_list):
    print(f"Loading {dataset} dataset")

    if dataset == "videomme":
        parquet_path = "dataset/videomme/test-00000-of-00001.parquet"
        df = pd.read_parquet(parquet_path)

        dataset = df[["question_id", "videoID", "question"]]

        query_list = []
        for data in dataset.values:
            if data[1] in video_list:
                query_list.append([data[0], data[1], data[2]])
            else:
                print(f"Video {data[1]} not in the video list")

    elif dataset == "nextqa":
        parquet_path = "dataset/nextqa/test-00000-of-00001.parquet"
        df = pd.read_parquet(parquet_path)

        dataset = df[["qid", "video", "question"]]

        query_list = []
        for data in dataset.values:
            if str(data[1]) in video_list:
                query_list.append([data[0], data[1], data[2]])

    elif dataset == "egoschema":
        parquet_path = "dataset/egoschema/test-00000-of-00001.parquet"
        df = pd.read_parquet(parquet_path)

        dataset = df[["question_idx", "video_idx", "question"]]

        query_list = []
        for data in dataset.values:
            if str(data[1]) in video_list:
                query_list.append([data[0], data[1], data[2]])

    elif dataset == "longvideobench":
        parquet_path = "dataset/longvideobench/validation-00000-of-00001.parquet"
        df = pd.read_parquet(parquet_path)

        dataset = df[["id", "video_path", "question"]]

        query_list = []
        for data in dataset.values:
            video_name = data[1].split(".")[0]
            if str(video_name) in video_list:
                query_list.append([data[0], video_name, data[2]])

    elif dataset == "mlvu":
        parquet_path = "dataset/mlvu/test-00000-of-00001.parquet"
        df = pd.read_parquet(parquet_path)

        dataset = df[["video_name", "video_name", "question"]]

        query_list = []
        for data in dataset.values:
            if str(data[1][:-4]) in video_list:
                query_list.append([data[0], data[1][:-4], data[2].split("\n")[0]])

    print(f"Total {len(query_list)} questions")
    return query_list


def inverse_transform_sampling(score, n, power=-1):
    # normalize the score to 0-1
    score = score - np.min(score)
    score = score / np.max(score)

    # power
    if power != -1:
        score = score**power

    # compute the cumulative distribution function (CDF)
    probabilities = score / np.sum(score)
    cdf = np.cumsum(probabilities)

    # generate uniform values between 0 and 1, exclude the 0 and 1 to avoid out of bounds
    uniform_sampling = np.linspace(1 / n, 1 - 1 / n, n)

    # use the inverse CDF to convert the uniform_sampling to indices
    sampled_indices = np.searchsorted(cdf, uniform_sampling)
    return sampled_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptively sample the frames based on similarity score.")
    parser.add_argument("--model", type=str, choices=["clip", "siglip"], default="clip")
    parser.add_argument("--feat_folder", type=str)
    parser.add_argument("--dataset", type=str, default="videomme")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--power", type=float, default=-1)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    # read video list from the feature folder
    video_list = sorted(os.listdir(args.feat_folder))
    video_list = [video.replace(".npy", "") for video in video_list]

    # read dataset annotation to get text query
    query_list = get_query_list(args.dataset, video_list)

    # build the CLIP model
    device = "cuda"  # the device to load the model onto
    model = get_model(args.model, device=device)

    video_dict = {}
    for question_id, video_name, question_query in tqdm.tqdm(query_list):

        # load the visual feature
        video_features = np.load(os.path.join(args.feat_folder, f"{video_name}.npy"))
        video_features = torch.from_numpy(video_features).to(device)  # [T,C]

        # load the text feature
        text_features = model.extract_text_features(question_query)

        # compute the similarity score
        score = model.compute_similarity(video_features, text_features).numpy()  # [T]

        # sample the frames
        frame_idxs = inverse_transform_sampling(score, args.num_frames, args.power)
        assert len(frame_idxs) == args.num_frames

        # convert the frame_idxs to time
        frame_seconds = frame_idxs / args.fps

        # save the moments
        tmp_result = {
            "question_id": question_id,
            "question": question_query,
            "frames": [round(f, 2) for f in frame_seconds.tolist()],
        }

        # save the result
        if video_name in video_dict:
            # some videos have multiple questions
            video_dict[video_name].append(tmp_result)
        else:
            video_dict[video_name] = [tmp_result]

    # save the json file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(video_dict, f)
