"""Trains frame ordering on audio snippets."""

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from act.model.modules.feature_extractors import (
    ResNetAudio,
)
from act.datasets.transforms import (
    AudioTimeCrop,
    AudioTimeCropDiscrete,
    AudioSpectrogram,
    AudioLog,
    AudioStandardNormalize,
    AudioUnsqueezeChannelDim,
)
from act.datasets.noise import Noise
from act.datasets.audioset import load_audioset

import warnings
warnings.filterwarnings("ignore")

# Set precision to make it efficient on DAS6 nodes
try:
    torch.set_float32_matmul_precision("medium")
except:
    pass


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


class OrderingModel(torch.nn.Module):

    def __init__(self, model, n_frames=5):
        super().__init__()
        self.feature_extractor = model
        n_perm = factorial(n_frames)
        self.ordering_head = torch.nn.Linear(512, n_perm)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.ordering_head(x)
        return x


class CODECTestAudioFrameOrdering(pl.LightningModule):
    def __init__(self, cfg, model, n_frames=5):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()        

    def get_batch_outputs(self, batch):
        inputs = batch["audio"]
        target = batch["targets"]["noise_target"] - 1
        if target.max() >= 50 or target.min() < 0:
            import ipdb; ipdb.set_trace()
        outputs = self.model(inputs)
        return {"logits": outputs, "target": target}

    def training_step(self, batch, batch_idx):
        outputs = self.get_batch_outputs(batch)
        loss = self.loss(outputs["logits"], outputs["target"])
        accu = (outputs["logits"].argmax(dim=1) == outputs["target"]).float().mean()
        self.log("train/loss", loss, sync_dist=True, prog_bar=True)
        self.log("train/accu", accu, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.get_batch_outputs(batch)
        loss = self.loss(outputs["logits"], outputs["target"])
        accu = (outputs["logits"].argmax(dim=1) == outputs["target"]).float().mean()
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/accu", accu, sync_dist=True)
        return outputs

    def configure_optimizers(self):
        """Configure optimizers."""
        lr = self.cfg["optimizer"]["lr"]
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-09,
        )
        return optimizer


if __name__ == "__main__":
    import argparse
    
    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="noise_codec", choices=["noise_codec"])
    parser.add_argument("--only_train", action="store_true")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--gpus", default=[0], nargs="+", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    # Load feature extractor
    n_snippets = 5
    n_classes = factorial(n_snippets)
    feat_ext = ResNetAudio(
        "resnet18", num_classes=n_classes, extract_features=True, flatten=True,
    )
    x = torch.randn(1, 1, 257, 400)
    y = feat_ext(x)
    assert y.shape == torch.Size([1, 512])

    # Define frame ordering model
    model = OrderingModel(feat_ext, n_frames=n_snippets)
    x = torch.randn(1, 1, 257, 400)
    y = model(x)
    assert y.shape == torch.Size([1, n_classes])
