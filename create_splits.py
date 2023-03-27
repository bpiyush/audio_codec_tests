"""Creates splits for noise data."""
import os
from os.path import join

from sklearn.model_selection import train_test_split

import shared.utils.io as io


if __name__ == "__main__":
    data_dir = "/ssd/pbagad/datasets/Noise/"
    split_dir = join(data_dir, "splits")
    audio_fps = 22050

    raw_dir = join(data_dir, f"raw@{audio_fps}")
    files = os.listdir(raw_dir)
    
    train_indices, test_indices = train_test_split(files, test_size=0.2, random_state=42)
    
    io.save_txt(train_indices, join(split_dir, "train.txt"))
    io.save_txt(test_indices, join(split_dir, "test.txt"))
