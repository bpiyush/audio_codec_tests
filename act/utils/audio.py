"""Audio utility functions."""
from pathlib import Path
import tempfile

import torch
import torchaudio
import ffmpeg


def reencode_aac(x, a_fps):
    # `with tempfile.TemporaryDirectory` creates a temporal directory in your TMPDIR and saves
    # encoded files there. Once the code goes out of `with` context, `tmp_dir` is deleted (convenient :) )
    with tempfile.TemporaryDirectory(suffix='python_ffmpeg_todel') as tmp_dir:
        # save tensor to the disk for further transcoding
        tmp_file_path_wav = Path(tmp_dir) / 'out.wav'
        torchaudio.save(tmp_file_path_wav, x, a_fps, compression=None, format='wav', encoding=None)
        # transcode (wav -> mp4 (aac))
        tmp_transcoded_path_mp4 = str(tmp_file_path_wav).replace('out.wav', 'out.mp4')
        process = (
            ffmpeg
            .input(tmp_file_path_wav)
            .output(tmp_transcoded_path_mp4, acodec='aac')
            .global_args('-loglevel', 'panic').global_args('-nostats').global_args('-hide_banner')
        )
        process, _ = process.run()
        # transcode (mp4 (aac) -> wav)
        tmp_transcoded_path_wav = str(tmp_transcoded_path_mp4).replace('out.mp4', 'out_pcm.wav')
        process = (
            ffmpeg
            .input(tmp_transcoded_path_mp4)
            .output(tmp_transcoded_path_wav, acodec='pcm_s16le')
            .global_args('-loglevel', 'panic').global_args('-nostats').global_args('-hide_banner')
        )
        process, _ = process.run()
        audio, _ = torchaudio.load(tmp_transcoded_path_wav)
    return audio


def get_video_and_audio(path, get_meta=False):
    # (T, 3, H, W) [0, 255, uint8] <- (T, H, W, 3)
    v_fps = 10
    a_fps = 22050
    clip_len_sec = 10

    # trying until a good sample is generated (sometimes it becomes shorter after reencoding)
    while True:
        rgb = torch.randint(0, 256, (v_fps*clip_len_sec, 3, 224, 224), dtype=torch.uint8)
        audio = torch.rand((1, a_fps*clip_len_sec), dtype=torch.float32) * 2 - 1

        # reencode with pcm_s16le and load with `torchaudio.load(.wav)`
        # audio = reencode_pcms16le(audio, a_fps)
        # reencode with aac and load with `torchaudio.load(.wav)` (`torchvision.io.read_video(.mp4)`)
        audio = reencode_aac(audio, a_fps)
        # reencode with opus and load with `torchaudio.load(.wav)`
        # audio = reencode_opus(audio, a_fps)
        # (Ta) <- (Ca, Ta)
        audio = audio.mean(dim=0)
        # trying to mask out the artifacts
        # audio += (torch.randn_like(audio) + 1e-5) * 1e-5

        if audio.shape[0] >= a_fps * (clip_len_sec - 1):
            break
        else:
            print(f'audio is short for some reason: {audio.shape}. Trying again')

    # FIXME: this is legacy format of `meta` as it used to be loaded by VideoReader.
    meta = {
        'video': {'fps': [v_fps]},
        'audio': {'framerate': [a_fps]},
    }
    return rgb, audio, meta


if __name__ == "__main__":
    path = "/tmp/sample.mp4"
    rgb, audio, meta = get_video_and_audio(path, get_meta=True)
    import ipdb; ipdb.set_trace()