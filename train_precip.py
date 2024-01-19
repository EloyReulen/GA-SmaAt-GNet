from root import ROOT_DIR

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
import argparse
import torch
from models import unet_precip_regression_lightning as unet_regr
import models.regression_GA_SmaAt_GNet as gan

def train_regression(hparams):
    if hparams.model == "SmaAt-UNet":
        net = unet_regr.SmaAt_UNet(hparams=hparams)
    elif hparams.model == "SmaAt-GNet":
        net = unet_regr.SmaAt_GNet(hparams=hparams)
    elif hparams.model == "SmaAt-GNet-Aleatoric":
        net = unet_regr.SmaAt_GNet_aleatoric(hparams=hparams)
    elif hparams.model == "GA-SmaAt-GNet":
        net = gan.GAN(hparams=hparams)
    else:
        raise Exception(f"{hparams.model} is not a valid model name")
    
    default_save_path = hparams.default_save_path

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshhold_50_{epoch}-{val_loss:.6f}",
        save_top_k=3,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )
    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)

    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "data" / "precipitation" / "train_test_1998-2022_input-length_12_img-ahead_12_rain-threshhold_50_normalized.h5",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args()

    # args.fast_dev_run = True
    args.n_channels = 12
    args.n_masks = 25
    args.n_classes = 12
    args.n_output_images = 12
    
    args.gpus = 1
    args.model = "GA-SmaAt-GNet" #SmaAt-UNet, SmaAt-GNet, SmaAt-GNet-Aleatoric or GA-SmaAt-GNet
    args.lr_patience = 4
    args.es_patience = 15
    # GAN options
    args.l = 1000000
    args.disc_every_n_steps = 2

    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.dataset_folder = (
        ROOT_DIR / "data" / "precipitation" / "train_test_1998-2022_input-length_12_img-ahead_12_rain-threshhold_50_normalized.h5"
    )
    args.dataset = "train"
    args.default_save_path = ROOT_DIR / "lightning" / "1998-2022" / f"{args.model}_batch-{args.batch_size}_v1.0"

    args.dropout=0.5
    # The default sharing strategy is not supported on mac
    torch.multiprocessing.set_sharing_strategy('file_system')

    # args.resume_from_checkpoint = f"lightning/precip_regression/[filename].ckpt"

    train_regression(args)

