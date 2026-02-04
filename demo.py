import argparse
import torch
import numpy as np
import decord
from extract_feature import get_model


def visualize(score, moments, query):
    # plot the score
    import matplotlib.pyplot as plt

    x = np.arange(len(score))
    plt.plot(x, score)
    plt.vlines(moments, ymin=0, ymax=1, colors="r")

    # plt.plot(score)
    plt.title(f"Query: {query}")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.savefig(f"demo.jpg")
    plt.close()
    print("Visualization saved to demo.jpg")


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
    return sampled_indices, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptively sample the frames based on similarity score.")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--query", type=str)
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--power", type=float, default=3)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for visual feature extraction")
    args = parser.parse_args()

    # build model
    device = "cuda"  # the device to load the model onto
    model = get_model("clip", device=device)

    # read the video
    video_reader = decord.VideoReader(args.video_path)
    video_fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)

    # change fps
    new_total_frames = int(total_frames / video_fps * args.fps)
    frame_idxs = np.arange(0, new_total_frames, 1)
    frame_idxs = np.round(frame_idxs / args.fps * video_fps)  # new index that mapped with original fps
    frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).astype(int)

    # extract visual features
    print("Extracting visual features...")
    feature_list = []
    for batch_start in range(0, len(frame_idxs), args.batch_size):
        # read frames
        batch_idxs = frame_idxs[batch_start : batch_start + args.batch_size]
        batch_images = video_reader.get_batch(batch_idxs).asnumpy()

        # extract visual features
        image_features = model.extract_visual_features(batch_images)
        feature_list.append(image_features.cpu().numpy())  # [T,C]

    video_features = torch.from_numpy(np.concatenate(feature_list, axis=0)).to(device)
    print(f"Video features shape: {video_features.shape}")

    # load the text feature
    text_features = model.extract_text_features(args.query)

    # compute the similarity score
    score = model.compute_similarity(video_features, text_features).numpy()  # [T]

    # sample the frames
    frame_idxs, score = inverse_transform_sampling(score, args.num_frames, args.power)
    assert len(frame_idxs) == args.num_frames

    # convert the frame_idxs to time
    frame_seconds = frame_idxs / args.fps
    print("Selected frames:\n", frame_seconds)

    # visualize the score
    visualize(score, frame_idxs, args.query)
