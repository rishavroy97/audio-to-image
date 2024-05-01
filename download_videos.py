import argparse
import concurrent.futures
import datetime
import os

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

import pandas as pd
import torch
from pytube import YouTube

TIME_DURATION_IN_SEC = 10
CSV_FILE = './vggsound.csv'

NEW_COLUMNS = {
    '---g-f_I2yQ': 'youtube_video_id',
    '1': 'start_seconds',
    'people marching': 'label',
    'test': 'split',
}


def download_video(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = './data/video'
    try:
        yt = YouTube(video_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            # Download the video
            stream.download(output_path, filename=f"full_vid_{video_id}.mp4")
    except Exception as e:
        print(f"Error downloading video {video_id}: {str(e)}")


def download_videos_from_dataframe(df, start, end):
    end = min(end, len(df))
    start = max(start, 0)
    video_ids = df.iloc[start: end + 1]['youtube_video_id'].tolist()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for video_id in video_ids:
            futures.append(executor.submit(download_video, video_id))
        for future in concurrent.futures.as_completed(futures):
            pass  # Wait for all downloads to complete


def main(start, end):
    df = pd.read_csv(CSV_FILE)
    df.rename(columns=NEW_COLUMNS, inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if device == "cpu" else "true"

    start_time = datetime.datetime.now()

    download_videos_from_dataframe(df, start, end)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Time taken for download: {duration:.6f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos from vggsound csv file.")
    parser.add_argument("--start", type=int, default=0, help="Starting index of dataframe (default: 0)")
    parser.add_argument("--end", type=int, default=-1, help="Starting index of dataframe (default: -1)")
    args = parser.parse_args()

    main(args.start, args.end)
