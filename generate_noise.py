"""Generates noise data with various encodings.
"""
import os
from os.path import join

import ffmpeg
import numpy as np
import torch
import torchaudio

import shared.utils.log as log


def generate_noise(audio_fps: int, audio_len: float):
    """Generates and saves N noise samples."""
    
    n_samples = int(audio_fps * audio_len)
    noise = torch.rand((1, n_samples), dtype=torch.float32) * 2 - 1

    return noise


def save_aac(raw_path, save_path):

    # 1. transcode (wav -> mp4 (aac))
    tmp_transcoded_path_mp4 = str(raw_path).replace('.wav', '.mp4')
    process = (
        ffmpeg
        .input(raw_path)
        .output(tmp_transcoded_path_mp4, acodec='aac')
        .global_args('-loglevel', 'panic').global_args('-nostats').global_args('-hide_banner')
    )
    process, _ = process.run()

    # 2. transcode (mp4 (aac) -> wav))
    process = (
        ffmpeg
        .input(tmp_transcoded_path_mp4)
        .output(save_path, acodec='pcm_s16le')
        .global_args('-loglevel', 'panic').global_args('-nostats').global_args('-hide_banner')
    )
    process, _ = process.run()
    
    # Remove the temporary file
    os.remove(tmp_transcoded_path_mp4)


if __name__ == "__main__":
    

    data_dir = "/ssd/pbagad/datasets/Noise/"
    audio_fps = 16000
    audio_len = 5.
    N = 1200

    raw_dir = join(data_dir, f"raw@{audio_fps}")
    os.makedirs(raw_dir, exist_ok=True)

    pcms16le_dir = join(data_dir, f"pcms16le@{audio_fps}")
    os.makedirs(pcms16le_dir, exist_ok=True)

    aac_dir = join(data_dir, f"aac@{audio_fps}")
    os.makedirs(aac_dir, exist_ok=True)


    iterator = log.tqdm_iterator(range(N), desc="Generating noise samples")
    for i in iterator:

        # Generate noise
        noise = generate_noise(audio_fps, audio_len)
        
        # Save raw noise
        raw_path = join(raw_dir, f"{i}.wav")
        torchaudio.save(
            raw_path, noise, audio_fps,
            compression=None, format='wav', encoding=None,
        )

        # Save PCM S16LE noise
        pcm_path = join(pcms16le_dir, f"{i}.wav")
        process = (
            ffmpeg
            .input(raw_path)
            .output(pcm_path, acodec='pcm_s16le')
            .global_args('-loglevel', 'panic')
            .global_args('-nostats')
            .global_args('-hide_banner')
        )
        process, _ = process.run()
        
        # Save AAC noise
        aac_path = join(aac_dir, f"{i}.wav")
        save_aac(raw_path, aac_path)
