import argparse
import concurrent.futures
import datetime
import os

import pandas as pd
import torch

TIME_DURATION_IN_SEC = 10
CSV_FILE = './vggsound.csv'

NEW_COLUMNS = {
    '---g-f_I2yQ': 'youtube_video_id',
    '1': 'start_seconds',
    'people marching': 'label',
    'test': 'split',
}


def delete_original_video_clip(video_id):
    input_video_path = './data/video/full_vid_' + video_id + '.mp4'
    audio_path = './data/audio/audio_' + video_id + '.wav'
    if os.path.exists(input_video_path) and os.path.exists(audio_path):
        os.remove(input_video_path)


def delete_vids_from_dataframe(df, start, end):
    end = min(end, len(df))
    start = max(start, 0)
    video_ids = df.iloc[start: end + 1]['youtube_video_id'].tolist()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(video_ids)):
            futures.append(executor.submit(delete_original_video_clip, video_ids[i]))
        for _ in concurrent.futures.as_completed(futures):
            pass  # Wait for all downloads to complete


def delete_vid(row):
    video_id = row['youtube_video_id']
    try:
        delete_original_video_clip(video_id)
        return True
    except Exception as e:
        return False


def delete_vids(df, start, end):
    end = min(end, len(df))
    start = max(start, 0)
    result = df.iloc[start:end + 1].apply(delete_vid, axis=1)
    return result


def main(start, end, is_concurrent):
    df = pd.read_csv(CSV_FILE)
    df.rename(columns=NEW_COLUMNS, inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if device == "cpu" else "true"

    start_time = datetime.datetime.now()

    if is_concurrent:
        delete_vids_from_dataframe(df, start, end)
    else:
        delete_vids(df, start, end)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Time taken for clipping: {duration:.6f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos from vggsound csv file.")
    parser.add_argument("--start", type=int, default=0, help="Starting index of dataframe (default: 0)")
    parser.add_argument("--end", type=int, default=-1, help="Starting index of dataframe (default: -1)")
    parser.add_argument('--concurrent', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(args.start, args.end, args.concurrent)
