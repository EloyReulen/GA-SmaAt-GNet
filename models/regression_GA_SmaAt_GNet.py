from root import ROOT_DIR
import lightning.pytorch as pl
from torch import nn, optim, multiprocessing
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
from utils import dataset_precip
from models.discriminator import *
from models.unet_precip_regression_lightning import SmaAt_UNet, SmaAt_GNet
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math


class GAN_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=12)
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.automatic_optimization = False

        # networks
        self.generator = SmaAt_GNet(hparams=hparams)
        self.discriminator = LargePix2PixDiscriminatorCBAM(hparams=hparams)

        self.g_losses = []
        self.d_losses = []
        self.log("val_g_loss",float("inf"),prog_bar=False)
        self.log("val_d_loss",float("inf"),prog_bar=False)

    def forward(self, x, m):
        return self.generator(x, m)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        scheduler_g = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt_g, mode="min", factor=0.1, patience=self.hparams.lr_patience,
                
            ),
            "monitor": "val_g_loss",  # Default: val_loss
        }
        scheduler_d = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt_d, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_d_loss",  # Default: val_loss
        }
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def on_validation_epoch_end(self):
        plt.close("all")
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def loss_func(self, y_pred, y_true):
        return nn.functional.mse_loss(
            y_pred, y_true, reduction="mean"
        )
    
    def training_step(self, batch, batchid):
        # lamda param for generator loss
        l = self.hparams.l

        imgs, masks_in, tar, masks_true = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()
        
        
        # how well can it label as real?
        valid = torch.ones((imgs.size(0),1,4,4))
        valid = valid.type_as(imgs)

        # how well can it label as fake?
        fake = torch.zeros((imgs.size(0),1,4,4))
        fake = fake.type_as(imgs)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        generated_imgs = self.generator(imgs, masks_in)
        self.toggle_optimizer(optimizer_d)
        
        real_loss = self.adversarial_loss(self.discriminator(imgs,tar), valid)
        fake_loss = self.adversarial_loss(self.discriminator(imgs, generated_imgs.detach()), fake)

        # discriminator loss is the sum of these
        d_loss = (real_loss + fake_loss)
        self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # only do backpropagation every n steps
        if batchid % self.hparams.disc_every_n_steps == 0:
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
        
        self.untoggle_optimizer(optimizer_d)

        # Train generator
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()

        generated_imgs = self(imgs, masks_in)

        # Generator loss is the combination of adverarial loss and MSE loss
        g_loss = self.adversarial_loss(self.discriminator(imgs,generated_imgs), valid)
        structural_g_loss = self.loss_func(generated_imgs, tar)
        total_g_loss = g_loss + l * structural_g_loss
        self.log("g_total", total_g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
        self.manual_backward(total_g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        if batchid % 100 == 0:
            display_list = [imgs[0], tar[0], generated_imgs[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']
            # Plot generated images
            f, axarr = plt.subplots(3,12)
            f.set_figwidth(20)
            for i in range(3):
                for j in range(12):
                    axarr[i, j].imshow(display_list[i][j,:,:].detach().cpu().numpy())
            plt.axis("off")
            plt.savefig(self.hparams.default_save_path / "imgs.png")

            # Plot losses
            xs = [x * 10 for x in range(len(self.g_losses))]
            xs = xs[10:]
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('steps')
            ax1.set_ylabel('g_loss', color=color)
            g_losses = self.g_losses[10:]
            ax1.plot(xs, g_losses, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('d_loss', color=color)  # we already handled the x-label with ax1
            d_losses = self.d_losses[10:]
            ax2.plot(xs, d_losses, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(self.hparams.default_save_path / "losses.png")

        
        # Save train losses for plotting
        if batchid > 0 and batchid % 10 == 0:
            self.g_losses.append(total_g_loss.item())
            self.d_losses.append(d_loss.item())

        #Update lr schedulers on end of epoch
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0 and self.trainer.current_epoch > 0:
            scheduler_d.step(self.trainer.callback_metrics["val_d_loss"])
            scheduler_g.step(self.trainer.callback_metrics["val_g_loss"])

        return total_g_loss

    def validation_step(self, batch, batch_idx):
        # lamda param for generator loss
        l = self.hparams.l

        imgs, masks_in, tar, masks_true = batch
        
        
        # how well can it label as real?
        valid = torch.ones((imgs.size(0),1,4,4))
        valid = valid.type_as(imgs)

        # how well can it label as fake?
        fake = torch.zeros((imgs.size(0),1,6,6))
        fake = torch.zeros((imgs.size(0),1,4,4))
        fake = fake.type_as(imgs)
        
        generated_imgs = self.generator(imgs, masks_in)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(imgs,tar), valid)
        fake_loss = self.adversarial_loss(self.discriminator(imgs, generated_imgs.detach()), fake)

        # discriminator loss is the sum of these
        d_loss = (real_loss + fake_loss)
        self.log("val_d_loss", d_loss, prog_bar=True)

        # Train generator
        # Generator loss is the combination of adverarial loss and MSE loss
        generated_imgs = self(imgs, masks_in)
        g_loss = self.adversarial_loss(self.discriminator(imgs,generated_imgs), valid)
        structural_g_loss = self.loss_func(generated_imgs, tar)
        
        total_g_loss = g_loss + l * structural_g_loss
        self.log("val_loss",structural_g_loss*100,prog_bar=True)
        self.log("val_g_loss",total_g_loss,prog_bar=True)


class GAN(GAN_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = GAN_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=12)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 12
        return parser

    def __init__(self, hparams):
        super(GAN, self).__init__(hparams=hparams)
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
