"""
Dataset of randomly generated audio-video noise.

Original author: Vladimir Iashin

"""
import os
import random
import sys
from pathlib import Path

import torch
import torchvision

from act.utils.audio import get_video_and_audio


class Noise(torch.utils.data.Dataset):
    
    def __init__(self, num_samples=10000, transforms=None):
        super().__init__()
        self.num_samples = num_samples
        self.transforms = transforms
    
    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(path, get_meta=True)
        return rgb, audio, meta
    
    def __getitem__(self, index):
        path = "/tmp/sample.mp4"
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
        return self.num_samples

    
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

    dataset = Noise(transforms=transforms)
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
    
