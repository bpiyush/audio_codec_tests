"""Codec noise dataset."""

import os
import random
import sys
from pathlib import Path
import pandas as pd

import torch
import torchvision
import torchaudio


class NoiseCodec(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        audio_len=5.,
        audio_fps=22050,
        codec="raw@22050",
        split="train",
        transforms=None,
    ):
        super().__init__()
        assert split in ["train", "test"]
        self.split = split
        self.transforms = transforms
        self.audio_len = audio_len
        self.audio_fps = audio_fps
        self.n_samples = int(audio_len * audio_fps)
        
        split_path = os.path.join(data_dir, "splits", f"{split}.txt")
        assert os.path.exists(split_path), f"Split file not found: {split_path}"
        
        audio_dir = os.path.join(data_dir, codec)
        assert os.path.isdir(audio_dir), f"Audio dir not found: {audio_dir}"
        
        audio_ids = [line.strip().split(".wav")[0] for line in open(split_path)]
        audio_paths = [os.path.join(audio_dir, f"{audio_id}.wav") for audio_id in audio_ids]
        self.df = pd.DataFrame({"audio_id": audio_ids, "audio_path": audio_paths})
    
    def __getitem__(self, index):

        item = self.df.iloc[index].to_dict()
        path = item["audio_path"]

        # Load audio
        audio, _ = torchaudio.load(path)

        # Convert to mono
        audio = audio.mean(dim=0)
        
        # Pad/cut if needed
        if audio.shape[0] < self.n_samples:
            audio = torch.cat(
                [audio, torch.zeros(self.n_samples - audio.shape[0])],
            )
        elif audio.shape[0] > self.n_samples:
            audio = audio[:self.n_samples]

        item["audio"] = audio

        # Placeholder (to add metadata) (needed for transforms)
        item["meta"] = {
            'audio': {},
        }

        # Apply transforms
        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.df)


def load_dataset(data_root, split, audio_fps, transforms, codec="raw", batch_size=16, num_workers=8, audio_len=5.):
    dataset = NoiseCodec(
        data_dir=os.path.join(data_root, "Noise"),
        codec=f"{codec}@{audio_fps}",
        audio_fps=audio_fps,
        audio_len=audio_len,
        split=split,
        transforms=transforms,
    )
    item = dataset[0]
    assert item["audio"].shape == torch.Size([1, 257, 862])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    batch = next(iter(dataloader))
    assert batch["audio"].shape == torch.Size([4, 1, 257, 862])
    return dataset, dataloader



if __name__ == "__main__":
    import act.datasets.transforms as audio_transforms
    transforms = [
        audio_transforms.AudioSpectrogram(n_fft=512, hop_length=128),
        audio_transforms.AudioLog(),
        audio_transforms.AudioStandardNormalize(),
        audio_transforms.AudioUnsqueezeChannelDim(dim=0),
    ]
    transforms = torchvision.transforms.Compose(transforms)
    audio_fps = 22050
    dataset = NoiseCodec(
        data_dir="/ssd/pbagad/datasets/Noise/",
        codec=f"raw@{audio_fps}",
        audio_fps=audio_fps,
        audio_len=5.,
        split="train",
        transforms=transforms,
    )
    item = dataset[0]
    print(item["audio"].shape)
    assert item["audio"].shape == torch.Size([1, 257, 862])
    
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
    )
    batch = next(iter(dataloader))
    assert batch["audio"].shape == torch.Size([4, 1, 257, 862])
