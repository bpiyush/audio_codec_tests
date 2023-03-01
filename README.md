# Audio Codec Tests
Tests for codec artefacts in stored audio samples.


## Setup

Create a `conda` environment as instructed in [`setup/steps.md`](setup/steps.md).

Make sure to activate the conda environment and also setting the right `PYTHONPATH`.
```sh
export PYTHONPATH=$PWD
```

## Data

Assuming you store all your datasets in a root folder `/path/to/datasets`, you can create a symlink to the dataset folder in this repo:
```bash
ln -s /path/to/datasets/ data
```

For example, this command for me is:
```sh
ln -s /ssd/pbagad/datasets/ data
```

The dataset folder structure should be as follows:
```sh
data
├── ACAV100M
├── activitynet-1.3
│   ├── annotations
│   ├── feat
│   ├── missing_files_v1-3_test
│   └── v1-3
├── AudioCaps
├── AudioSet
│   ├── annotations
│   ├── clips
│   ├── cut_clips
│   ├── strong_annotations
│   └── videos
:   :
└── VindLU-Data
    ├── anno_downstream
    └── videos_images

143 directories
```
Note that for now, we only need the `AudioSet` dataset.


## Models

For now, we stick to using `resnet18` as the base model.

## Training

* Training on dataset of random noise encoded with `aac@22050`:
```sh
python train.py --no_wandb --dataset noise --batch_size 4 --epochs 100
```
Note that the `--no_wandb` flag is used to disable logging to `wandb`. You can remove this flag if you want to log to `wandb`.

* Training on AudioSet dataset encoded with `aac@22050`:
```sh
python train.py --no_wandb --dataset audioset --batch_size 4 --epochs 100
```
