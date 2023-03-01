"""Main trainer script."""

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


class CODECTestAudio(pl.LightningModule):
    def __init__(self, cfg, model):
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
    parser.add_argument("--dataset", type=str, default="noise", choices=["noise", "audioset"])
    parser.add_argument("--only_train", action="store_true")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--gpus", default=[0], nargs="+", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--time_crop",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
        help="Whether to use continuous (e.g. 2.34, 2.45, ...)"\
            "or discrete (2.3, 2.4,..) time cropping."
    )
    args = parser.parse_args()

    # Load model
    model = ResNetAudio("resnet18", num_classes=50, extract_features=False)

    # Load dataset
    if args.time_crop == "continuous":
        TimeCrop = AudioTimeCrop
    elif args.time_crop == "discrete":
        TimeCrop = AudioTimeCropDiscrete
    else:
        raise ValueError(f"Invalid time_crop: {args.time_crop}")
    transforms = [
        TimeCrop(crop_len_sec=5.0, is_random=True),
        AudioSpectrogram(n_fft=512, hop_length=128),
        AudioLog(),
        AudioStandardNormalize(),
        AudioUnsqueezeChannelDim(dim=0),
    ]
    transforms = torchvision.transforms.Compose(transforms)

    if args.dataset == "noise":
        train_dataset = Noise(num_samples=10000, transforms=transforms)
        valid_dataset = Noise(num_samples=1000, transforms=transforms)
    elif args.dataset == "audioset":
        train_dataset = load_audioset(transforms=transforms, split="train")
        valid_dataset = load_audioset(transforms=transforms, split="val")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
        
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    
    cfg = {
        "optimizer": {"lr": args.lr},
    }
    pl_module = CODECTestAudio(cfg, model)
    
    # Define W&B logger
    if args.no_wandb:
        logger = None
    else:
        run_name = f"audio_coded_tests-resnet18-"\
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

    # Run validation
    if not args.only_train:
        trainer.validate(
            pl_module,
            dataloaders=valid_dataloader,
        )

    # Run training
    if not args.only_eval:
        trainer.fit(
            pl_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=train_dataloader,
        )
