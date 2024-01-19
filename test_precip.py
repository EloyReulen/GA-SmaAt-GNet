import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pickle
from tqdm import tqdm
import math
from pathlib import Path

from root import ROOT_DIR
from utils import dataset_precip
from models import unet_precip_regression_lightning as unet_regr
import models.regression_GA_SmaAt_GNet as gan


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def get_metrics(model, model_name, test_dl, denormalize=True, threshold=0.5, k=10):
    with torch.no_grad():
      mps = torch.device("mps")
      if model_name != "Persistence":
        model.eval()  # or model.freeze()?
        model.apply(apply_dropout)
        model.to(mps)
      loss_func = nn.functional.mse_loss
      
      factor = 1
      if denormalize:
          factor = 52.52

      threshold = threshold
      epsilon = 1e-6

      total_tp = 0
      total_fp = 0
      total_tn = 0
      total_fn = 0

      loss_denorm = 0.0
      f1 = 0.0
      csi = 0.0
      uncertainty = 0.0
      count = 0
      for x, mask, y_true, _ in tqdm(test_dl, leave=False):
          count += 1
          x = x.to(mps)
          mask = mask.to(mps)
          y_true = y_true.to(mps).squeeze()
          y_true = y_true
          y_pred = None
          y_preds = []

          if model_name == "Persistence":
            y_pred = x.squeeze()[11].repeat(12, 1, 1)
            uncertainty += 0
          else:
            for _ in range(k):
              if model_name == "SmaAt-UNet":
                pred = model(x)
              else:
                pred = model(x,mask)
              y_preds.append(pred.squeeze())
            y_preds = torch.stack(y_preds, dim=0)
            y_pred = torch.mean(y_preds, dim=0)
            uncertainty += torch.mean(torch.var(y_preds, dim=0)).item()
          
          # denormalize
          y_pred_adj = y_pred * factor
          y_true_adj = y_true * factor
          # calculate loss on denormalized data
          loss_denorm += loss_func(y_pred_adj, y_true_adj, reduction="sum")
          # sum all output frames
          y_pred_adj = torch.sum(y_pred_adj, axis=0)
          y_true_adj = torch.sum(y_true_adj, axis=0)
          # convert to masks for comparison
          y_pred_mask = y_pred_adj > threshold
          y_true_mask = y_true_adj > threshold
          y_pred_mask = y_pred_mask.cpu()
          y_true_mask = y_true_mask.cpu()

          tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
          total_tp += tp
          total_fp += fp
          total_tn += tn
          total_fn += fn

      uncertainty /= len(test_dl)
      mse_image = loss_denorm / len(test_dl)
      mse_pixel = mse_image / torch.numel(y_true)
      # get metrics
      precision = total_tp / (total_tp + total_fp + epsilon)
      recall = total_tp / (total_tp + total_fn + epsilon)
      f1 = 2 * precision * recall / (precision + recall + epsilon)
      csi = total_tp / (total_tp + total_fn + total_fp + epsilon)
      hss = (total_tp * total_tn - total_fn * total_fp) / ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn) + epsilon)
      mcc = calculate_mcc(total_tp, total_tn, total_fp, total_fn)
    return mse_pixel.item(), f1, csi, hss, mcc, uncertainty

def calculate_mcc(total_tp, total_tn, total_fp, total_fn):
    total_tp = np.array(total_tp, dtype=np.float64)
    total_tn = np.array(total_tn, dtype=np.float64)
    total_fp = np.array(total_fp, dtype=np.float64)
    total_fn = np.array(total_fn, dtype=np.float64)

    numerator = (total_tp * total_tn) - (total_fp * total_fn)
    denominator = np.sqrt((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
    mcc = numerator / denominator if denominator != 0 else 0
    return mcc


def get_model_losses(model_file, model_name, data_file, denormalize):
    test_losses = dict()
    dataset = dataset_precip.precipitation_maps_masked_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=12, 
        mode="test")

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # load the model
    if model_name == "SmaAt-UNet":
      model = unet_regr.SmaAt_UNet
      model = model.load_from_checkpoint(f"{model_file}")
    elif model_name == "SmaAt-GNet":
      model = unet_regr.SmaAt_GNet
      model = model.load_from_checkpoint(f"{model_file}")
    elif model_name == "GA-SmaAt-GNet":
      model = gan.GAN
      model = model.load_from_checkpoint(f"{model_file}")
    elif model_name == "Persistence":
      model = None
    else:
      raise Exception(f"{model_name} is not a valid model name")
    
    thresholds = [0.5, 10, 20]
    for threshold in thresholds:
      print(str(int(threshold*100)))
      test_losses[f"binary_{str(int(threshold*100))}"] = []

    for threshold in thresholds:
      losses = get_metrics(model, model_name, test_dl, denormalize, threshold=threshold, k=10)
      test_losses[f"binary_{str(int(threshold*100))}"].append([threshold, model_name] + list(losses))

        
    return test_losses

def losses_to_csv(losses, path):
    csv = "threshold, name, mse, f1, csi, hss, mcc, uncertainty\n"
    for loss in losses:
      row = ",".join(str(l) for l in loss)
      csv += row + "\n"

    with open(path,"w+") as f:
      f.write(csv)

    return csv


if __name__ == '__main__':
    denormalize = True
    data_file = (
        ROOT_DIR / "data" / "precipitation" / "train_test_1998-2022_input-length_12_img-ahead_12_rain-threshhold_50_normalized.h5"
    )
    results_folder = ROOT_DIR / "results"
    
    model_file = ROOT_DIR / "checkpoints" / "top_models/GA-SmaAt-GNet_rain_threshhold_50_epoch=26-val_loss=0.000288.ckpt"
    model_name = "GA-SmaAt-GNet" #Persistence, SmaAt-UNet, SmaAt-GNet or GA-SmaAt-GNet

    test_losses = get_model_losses(model_file, model_name, data_file, denormalize)

    print(losses_to_csv(test_losses['binary_50'], (results_folder / f"{model_name}_res_50.csv")))
    print(losses_to_csv(test_losses['binary_1000'], (results_folder / f"{model_name}_res_1000.csv")))
    print(losses_to_csv(test_losses['binary_2000'], (results_folder / f"{model_name}_res_2000.csv")))

