"""Trains frame ordering on audio snippets."""
import itertools

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
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
    def __init__(self, cfg, model, n_frames=5, K=4):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.n_frames = n_frames
        self.K = K
        indices = np.arange(self.n_frames)
        self.perms = np.array(list(itertools.permutations(indices)))

    def get_batch_outputs(self, batch):
        inputs = batch["audio"]
        batch_size = inputs.shape[0]
        
        # sample K permutations of the frames
        sample_from = list(range(len(self.perms)))
        idxs = np.random.choice(sample_from, self.K, replace=False)
        perms = self.perms[idxs]

        # Iterate over class label index, permutation
        all_inputs = []
        all_target = []
        for i, perm in zip(idxs, perms):
            T = inputs.shape[-1]
            t_indices = np.arange(T)
            t_snippets = np.array_split(t_indices, self.n_frames)
            t_snippets = [x[:-2] for x in t_snippets]
            t_indices = np.concatenate(np.array(t_snippets)[perm])
            inputs_shuffled = inputs[..., t_indices]
            # pad the audio to the original length along dim=-1
            inputs_shuffled = F.pad(inputs_shuffled, (0, T - inputs_shuffled.shape[-1]))
            all_inputs.append(inputs_shuffled)
            # Add targets
            all_target.extend([i] * batch_size)
        all_inputs = torch.cat(all_inputs, dim=0)
        target = torch.tensor(all_target).long().to(all_inputs.device)
        logits = self.model(all_inputs)
        return {"logits": logits, "target": target}

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
    
    if args.only_eval:
        args.gpus = [0]
    
    data_root = "/ssd/pbagad/datasets/"

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

    # Define dataset and dataloader
    from act.datasets.noise_codec import NoiseCodec, load_dataset
    import act.datasets.transforms as audio_transforms
    transforms = [
        audio_transforms.AudioSpectrogram(n_fft=512, hop_length=128),
        audio_transforms.AudioLog(),
        audio_transforms.AudioStandardNormalize(),
        audio_transforms.AudioUnsqueezeChannelDim(dim=0),
    ]
    transforms = torchvision.transforms.Compose(transforms)
    audio_fps = 22050
    train_ds, train_dl = load_dataset(
        data_root, "train", audio_fps, transforms,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    valid_ds, valid_dl = load_dataset(
        data_root, "test", audio_fps, transforms,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    
    cfg = {
        "optimizer": {"lr": args.lr},
    }
    pl_module = CODECTestAudioFrameOrdering(cfg, model)

    # Define W&B logger
    if args.no_wandb:
        logger = None
    else:
        run_name = f"act-frame-ordering-resnet18-"\
            f"lr-{args.lr}-bs-{args.batch_size}"
        if args.suffix:
            run_name += f"-{args.suffix}"
        logger = pl.loggers.WandbLogger(
            project="audio-visual",
            entity="bpiyush",
            name=run_name,
        )

    # Define trainer
    trainer = pl.Trainer(
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        strategy = DDPStrategy(find_unused_parameters=False),
    )

    if args.ckpt_path is not None:
        print(f">>> Initializing with checkpoint: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
        pl_module.load_state_dict(state_dict)

    # Run validation
    if not args.only_train:
        trainer.validate(
            pl_module,
            dataloaders=valid_dl,
        )

    # Run training
    if not args.only_eval:
        trainer.fit(
            pl_module,
            train_dataloaders=train_dl,
            val_dataloaders=valid_dl,
        )
