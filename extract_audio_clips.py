import argparse
import concurrent.futures
import datetime
import os

import pandas as pd
import torch
from moviepy.editor import VideoFileClip

TIME_DURATION_IN_SEC = 10
CSV_FILE = './vggsound.csv'

NEW_COLUMNS = {
    '---g-f_I2yQ': 'youtube_video_id',
    '1': 'start_seconds',
    'people marching': 'label',
    'test': 'split',
}


def extract_audio_clip(video_id, start_time):
    input_video_path = './data/video/full_vid_' + video_id + '.mp4'
    if not os.path.exists(input_video_path):
        return

    try:
        video_clip = VideoFileClip(input_video_path)

        if video_clip.duration < start_time + TIME_DURATION_IN_SEC:
            raise Exception(f'{video_id} - audio clip too short, skipping')

        # Define the subclip with the specified start and end times
        subclip = video_clip.subclip(start_time, start_time + TIME_DURATION_IN_SEC)

        # Extract audio from the subclip
        audio_clip = subclip.audio

        # Resample the audio to 16 kHz
        resampled_audio_clip = audio_clip.set_fps(16000)

        # Write the resampled audio to a WAV file
        audio_file_path = f"./data/audio/audio_{video_id}.wav"
        resampled_audio_clip.write_audiofile(audio_file_path, codec='pcm_s16le', verbose=False, logger=None)

        # Close the clips
        video_clip.close()
        audio_clip.close()

    except Exception as e:
        print(f"Error extracting audio clip: {e}")


def extract_clip(row):
    video_id = row['youtube_video_id']
    start_time = row['start_seconds']
    try:
        extract_audio_clip(video_id, start_time)
        return True
    except Exception as e:
        return False


def extract_clips(df, start, end):
    end = min(end, len(df))
    start = max(start, 0)
    result = df.iloc[start:end + 1].apply(extract_clip, axis=1)
    return result


def extract_audio_from_dataframe(df, start, end):
    end = min(end, len(df))
    start = max(start, 0)
    video_ids = df.iloc[start: end + 1]['youtube_video_id'].tolist()
    start_times = df.iloc[start: end + 1]['start_seconds'].tolist()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(video_ids)):
            futures.append(executor.submit(extract_audio_clip, video_ids[i], start_times[i]))
        for _ in concurrent.futures.as_completed(futures):
            pass  # Wait for all downloads to complete


def main(start, end):
    df = pd.read_csv(CSV_FILE)
    df.rename(columns=NEW_COLUMNS, inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if device == "cpu" else "true"

    start_time = datetime.datetime.now()

    # parallel processing code
    extract_audio_from_dataframe(df, start, end)
    # extract_clips(df, start, end)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Time taken for clipping: {duration:.6f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio/video clips from YouTube videos")
    parser.add_argument("--start", type=int, default=0, help="Starting index of dataframe (default: 0)")
    parser.add_argument("--end", type=int, default=-1, help="Starting index of dataframe (default: -1)")
    args = parser.parse_args()

    main(args.start, args.end)
