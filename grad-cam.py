from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import torch.nn as nn
from utils import dataset_precip
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import models.regression_GA_SmaAt_GNet as gan
from root import ROOT_DIR

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class SemanticSegmentationTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if device == 'cuda':
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def load_model(model, model_folder, device):
    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    model_file = models[-1]
    model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
    model.eval()
    model.to(torch.device(device))
    return model


def get_segmentation_data():
    dataset_masked = dataset_precip.precipitation_maps_masked_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=12, 
        mode="test",
        use_timestamps=False)

    test_dl_masked = torch.utils.data.DataLoader(
        dataset_masked,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return test_dl_masked


def run_cam(model, target_layers, device):
    test_dl = get_segmentation_data()
    count = 0
    for x, masks, y_true, _ in tqdm(test_dl, leave=False):
        count += 1
        if count < 13871:
            continue
        x = x.to(torch.device(device))
        masks = masks.to(torch.device(device))
        model = model.to(torch.device(device))
        input = torch.cat((x,masks),dim=1)
        output = model(input)

        x = torch.sum(x, dim=1)
        output = torch.sum(output, dim=1)

        mask = np.digitize((output[0] * 52.52).detach().cpu().numpy(), np.array([0.5]), right=True)
        mask_float = np.float32(mask)
        image = torch.stack([x[0], x[0], x[0]], dim=2)
        image = image.cpu().numpy()
        targets = [SemanticSegmentationTarget(0, mask_float, device)]
        cam_image = []

        for layer in target_layers:
            with GradCAM(model=model, target_layers=layer) as cam:
                grayscale_cam = cam(input_tensor=input, targets=targets)[0, :]
                cam_image.append(show_cam_on_image(image, grayscale_cam, use_rgb=True))

        # Plot encoder
        fig, axes = plt.subplots(5,4, figsize=(8,10))
        # map encoder
        # 0
        axes[0][0].imshow(cam_image[0])
        axes[0][0].axis("off")
        axes[0][1].imshow(cam_image[1])
        axes[0][1].axis("off")
        # 1
        axes[1][0].imshow(cam_image[2])
        axes[1][0].axis("off")
        axes[1][1].imshow(cam_image[3])
        axes[1][1].axis("off")
        # 2
        axes[2][0].imshow(cam_image[4])
        axes[2][0].axis("off")
        axes[2][1].imshow(cam_image[5])
        axes[2][1].axis("off")
        # 3
        axes[3][0].imshow(cam_image[6])
        axes[3][0].axis("off")
        axes[3][1].imshow(cam_image[7])
        axes[3][1].axis("off")
        # 4
        axes[4][0].imshow(cam_image[8])
        axes[4][0].axis("off")
        axes[4][1].imshow(cam_image[9])
        axes[4][1].axis("off")

        # mask encoder
        # 0
        axes[0][2].imshow(cam_image[10])
        axes[0][2].axis("off")
        axes[0][3].imshow(cam_image[11])
        axes[0][3].axis("off")
        # 1
        axes[1][2].imshow(cam_image[12])
        axes[1][2].axis("off")
        axes[1][3].imshow(cam_image[13])
        axes[1][3].axis("off")
        # 2
        axes[2][2].imshow(cam_image[14])
        axes[2][2].axis("off")
        axes[2][3].imshow(cam_image[15])
        axes[2][3].axis("off")
        # 3
        axes[3][2].imshow(cam_image[16])
        axes[3][2].axis("off")
        axes[3][3].imshow(cam_image[17])
        axes[3][3].axis("off")
        # 4
        axes[4][2].imshow(cam_image[18])
        axes[4][2].axis("off")
        axes[4][3].imshow(cam_image[19])
        axes[4][3].axis("off")
        plt.tight_layout()
        plt.savefig("imgs/enc_grad.png")
        plt.show()

        # Plot decoder
        fig, axes = plt.subplots(4,1, figsize=(2,7))
        # 0
        axes[0].imshow(cam_image[20])
        axes[0].axis("off")
        # 1
        axes[1].imshow(cam_image[21])
        axes[1].axis("off")
        # 2
        axes[2].imshow(cam_image[22])
        axes[2].axis("off")
        # 3
        axes[3].imshow(cam_image[22])
        axes[3].axis("off")
        plt.tight_layout()
        plt.savefig("imgs/dec_grad.png")
        plt.show()
        
# Wrapper for GA-SmaAt-GNet that splits 1 input into 2 because gradcam needs 1 unput
class GradCAMWrapper(nn.Module):
    def __init__(self, model):
        super(GradCAMWrapper, self).__init__()
        self.model = model

    def forward(self, input_tensor):
        # Split the input tensor into multiple tensors
        input_tensor1 = input_tensor[:,:12]
        input_tensor2 = input_tensor[:,12:]

        # Forward pass through the original model
        output = self.model(input_tensor1, input_tensor2)

        return output


if __name__ == '__main__':
    data_file = (
        ROOT_DIR / "data" / "precipitation" / "train_test_1998-2022_input-length_12_img-ahead_12_rain-threshhold_50_normalized.h5"
    )
    device = 'cpu'
    # load the models
    ga_smaat_gnet = gan.GAN
    ga_smaat_gnet = ga_smaat_gnet.load_from_checkpoint("checkpoints/top_models/GA-SmaAt-GNet_rain_threshhold_50_epoch=26-val_loss=0.000288.ckpt")
    model = GradCAMWrapper(ga_smaat_gnet)
    print(model)
    target_layers = [
        [model.model.generator.inc1], [model.model.generator.cbam11],
        [model.model.generator.down11.maxpool_conv[1]], [model.model.generator.cbam12],
        [model.model.generator.down12.maxpool_conv[1]], [model.model.generator.cbam13],
        [model.model.generator.down13.maxpool_conv[1]], [model.model.generator.cbam14],
        [model.model.generator.down14.maxpool_conv[1]], [model.model.generator.cbam15],
        [model.model.generator.inc2], [model.model.generator.cbam21],
        [model.model.generator.down21.maxpool_conv[1]], [model.model.generator.cbam22],
        [model.model.generator.down22.maxpool_conv[1]], [model.model.generator.cbam23],
        [model.model.generator.down23.maxpool_conv[1]], [model.model.generator.cbam24],
        [model.model.generator.down24.maxpool_conv[1]], [model.model.generator.cbam25],
        [model.model.generator.up4.conv],
        [model.model.generator.up3.conv],
        [model.model.generator.up2.conv],
        [model.model.generator.up1.conv],
    ]
    run_cam(model, target_layers, device)