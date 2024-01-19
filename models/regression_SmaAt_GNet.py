import lightning.pytorch as pl
from torch import nn, optim, multiprocessing
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from utils import dataset_precip
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math


class UNet_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=1)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_loss",  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        return nn.functional.mse_loss(
            y_pred, y_true, reduction="mean"
        )
    
    def training_step(self, batch, batchid):
        x, mask, y, _ = batch
        y_pred = self(x, mask)

        if batchid % 100 == 0:
            # log sampled images
            display_list = [x[0], y[0], y_pred[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']

            f, axarr = plt.subplots(3,12)
            f.set_figwidth(20)
            for i in range(3):
                for j in range(12):
                    axarr[i, j].imshow(display_list[i][j,:,:].detach().cpu().numpy())
            plt.axis("off")
            plt.savefig(self.hparams.default_save_path / "imgs.png")
        loss = self.loss_func(y_pred.squeeze(), y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # *100 for readability in progress bar
        self.log("train_loss", loss*100, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y, _ = batch
        y_pred = self(x, mask)
        loss = self.loss_func(y_pred.squeeze(), y)
        self.log("val_loss", loss * 100, prog_bar=True)
    
    def on_validation_epoch_end(self):
        plt.close("all")

class Precip_regression_base_gnet(UNet_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNet_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=12)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 12
        return parser

    def __init__(self, hparams):
        super(Precip_regression_base_gnet, self).__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip()]
        # )
        train_transform = None
        valid_transform = None
        precip_dataset = dataset_precip.precipitation_maps_masked_h5
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            mode=self.hparams.dataset,
            transform=train_transform,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,
            num_output_images=self.hparams.num_output_images,
            mode=self.hparams.dataset,
            transform=valid_transform,
        )

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.seed(123)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):

        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            num_workers=0,
            pin_memory=True,
        )
        return valid_loader
