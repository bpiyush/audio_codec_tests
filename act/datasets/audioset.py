"""AudioSet dataset."""
import os
from os.path import join, exists
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision

from act.utils.audio import get_video_and_audio
from act.utils.log import repo_path, print_update


def load_csv(csv_path, video_dir, load="clip", video_id_key="# YTID", verbose=True):
    """Loads data from a given split file."""

    assert load in ["clip", "video"]

    df = pd.read_csv(csv_path)

    # add video (clip) paths
    if load == "clip":
        df["clip_path"] = df[[video_id_key, "start_seconds", "end_seconds_new"]].apply(
            lambda x: join(video_dir, f"{x[0]}_{float(x[1])}_{float(x[2])}.mp4"), axis=1,
        ).copy()
    elif load == "video":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown load: {load}")
    
    # filter rows that have clip present
    df = df[df["clip_path"].apply(exists)]
    
    if verbose:
        print_update(f"Loaded {len(df)} rows from {csv_path}")
    
    return df


class AudioSet(torch.utils.data.Dataset):
    
    def __init__(self, media_dir, csv_path, ext=".mp4", transforms=None):
        super().__init__()

        self.media_dir = media_dir
        self.csv_path = csv_path
        self.ext = ext

        # Load csv
        self.df = load_csv(
            self.csv_path,
            self.media_dir,
            video_id_key="video_id",
            load="clip",
            verbose=True,
        )
        
        self.transforms = transforms
    
    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(path, get_meta=True)
        return rgb, audio, meta
    
    def __getitem__(self, index):
        row = self.df.iloc[index].to_dict()
        path = row["clip_path"]
        rgb, audio, meta = self.load_media(path)
        item = self.make_datapoint(path, rgb, audio, meta)
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def make_datapoint(self, path, rgb, audio, meta):
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        # target = self.video2target[Path(path).stem[:11]]
        item = {
            'video': rgb,
            'audio': audio,
            'meta': meta,
            'path': str(path),
            'targets': {'a_start_i_sec': 0, 'noise_target': None},
            # 'targets': {'vggsound_target': target, 'vggsound_label': self.target2label[target]},
            # 'split': self.split,
        }

        # # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        # if self.load_fixed_offsets_on_test and self.split in ['valid', 'test']:
        #     item['targets']['offset_sec'] = self.vid2offset_params[Path(path).stem]['offset_sec']
        #     item['targets']['v_start_i_sec'] = self.vid2offset_params[Path(path).stem]['v_start_i_sec']

        return item
    
    def __len__(self):
        return len(self.df)

    
if __name__ == "__main__":
    import time
    

    from act.datasets.transforms import (
        # AudioTimeCrop,
        AudioTimeCropDiscrete,
        AudioSpectrogram,
        AudioLog,
        AudioStandardNormalize,
        AudioUnsqueezeChannelDim,
    )
    
    transforms = [
        AudioTimeCropDiscrete(crop_len_sec=5.0, is_random=True),
        AudioSpectrogram(n_fft=512, hop_length=128),
        AudioLog(),
        AudioStandardNormalize(),
        AudioUnsqueezeChannelDim(dim=0),
    ]
    transforms = torchvision.transforms.Compose(transforms)

    data_root = os.path.join(repo_path, "data")
    data_dir = os.path.join(data_root, "AudioSet")
    csv_path = os.path.join(data_dir, "annotations/clean_eval_segments-v1-val.csv")
    video_dir = os.path.join(data_dir, "cut_clips")

    dataset = AudioSet(
        media_dir=video_dir,
        csv_path=csv_path,
        transforms=transforms,
    )
    item = dataset[0]
    print("Start time: ", item["meta"]["start_sec"])
    print("Label: ", item['targets']['noise_target'])
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    start = time.time()
    batch = next(iter(dataloader))
    end = time.time()
    print("Time taken to create a single batch: ", end - start)
    print(batch.keys())
    print(batch['video'].shape)
    print(batch['audio'].shape)
    print(len(batch["targets"]["noise_target"]), batch["targets"]["noise_target"])
    
