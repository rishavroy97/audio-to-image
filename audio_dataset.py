import os.path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

NEW_COLUMN_NAMES = {
    '---g-f_I2yQ': 'youtube_video_id',
    '1': 'start_seconds',
    'people marching': 'label',
    'test': 'split',
}


class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.audio_dir = audio_dir
        self.df = pd.read_csv(csv_file)
        self.rename_columns()
        self.add_columns()
        self.remove_invalid_rows()

    @staticmethod
    def check_validity(file_path):
        return os.path.isfile(file_path)

    @staticmethod
    def transform_waveform(waveform, sampling_rate):
        new_frequency = 16000
        transform = torchaudio.transforms.Resample(sampling_rate, new_frequency)
        waveform = transform(waveform)
        mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
        return mono_waveform, new_frequency

    def remove_invalid_rows(self):
        self.df['is_valid'] = self.df['audio_path'].apply(AudioDataset.check_validity)
        self.df = self.df[self.df['is_valid'] == True]
        self.df = self.df.drop(columns=['is_valid'])

    def rename_columns(self):
        self.df.rename(columns=NEW_COLUMN_NAMES, inplace=True)

    def add_columns(self):
        self.df['audio_path'] = self.df['youtube_video_id'].apply(
            lambda x: self.audio_dir + '/' + 'audio_' + x + '.wav')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]['label']
        audio_path = self.df.iloc[idx]['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path, channels_first=True)
        waveform, sample_rate = AudioDataset.transform_waveform(waveform, sample_rate)
        return waveform, label

    def get_df(self):
        return self.df
