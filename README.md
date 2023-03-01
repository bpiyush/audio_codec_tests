# Audio Codec Tests
Tests for codec artefacts in stored audio samples.


## Setup

Create a `conda` environment as instructed in [`setup/steps.md`](setup/steps.md).

## Data

Assuming you store all your datasets in a root folder `/path/to/datasets`, you can create a symlink to the dataset folder in this repo:
```bash
ln -s /path/to/datasets/ data
```

For example, this command for me is:
```sh
ln -s /ssd/pbagad/datasets/ data
```