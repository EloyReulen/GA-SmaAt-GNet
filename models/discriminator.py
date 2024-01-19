import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.layers import CBAM 
    

# Base code obtained from https://github.com/togheppi/pix2pix/blob/master/model.py
class LargePix2PixDiscriminatorCBAM(nn.Module):
    # initializers
    def __init__(self, hparams, in_channels=24, d=64):
        super(LargePix2PixDiscriminatorCBAM, self).__init__()
        reduction_ratio = 16
        self.conv1 = nn.Conv2d(in_channels, d, 4, 1, 1)
        self.cbam1 = CBAM(d, reduction_ratio=reduction_ratio)
        self.conv_down_1 = nn.Conv2d(d, d, 4, 2, 1)
        self.cbam_down_1 = CBAM(d, reduction_ratio=reduction_ratio)

        self.conv2 = nn.Conv2d(d, d, 4, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d)
        self.cbam2 = CBAM(d, reduction_ratio=reduction_ratio)
        self.conv_down_2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv_down_2_bn = nn.BatchNorm2d(d * 2)
        self.cbam_down_2 = CBAM(d * 2, reduction_ratio=reduction_ratio)
        
        self.conv3 = nn.Conv2d(d * 2, d * 2, 4, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 2)
        self.cbam3 = CBAM(d * 2, reduction_ratio=reduction_ratio)
        self.conv_down_3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv_down_3_bn = nn.BatchNorm2d(d * 4)
        self.cbam_down_3 = CBAM(d * 4, reduction_ratio=reduction_ratio)
        
        self.conv4 = nn.Conv2d(d * 4, d * 4, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 4)
        self.cbam4 = CBAM(d * 4, reduction_ratio=reduction_ratio)
        self.conv_down_4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv_down_4_bn = nn.BatchNorm2d(d * 8)
        self.cbam_down_4 = CBAM(d * 8, reduction_ratio=reduction_ratio)
        
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.cbam1(x)
        x = F.leaky_relu(self.conv_down_1(x), 0.2)
        x = self.cbam_down_1(x)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.cbam2(x)
        x = F.leaky_relu(self.conv_down_2_bn(self.conv_down_2(x)), 0.2)
        x = self.cbam_down_2(x)
        
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.cbam3(x)
        x = F.leaky_relu(self.conv_down_3_bn(self.conv_down_3(x)), 0.2)
        x = self.cbam_down_3(x)

        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.cbam4(x)
        x = F.leaky_relu(self.conv_down_4_bn(self.conv_down_4(x)), 0.2)
        x = self.cbam_down_4(x)

        x = F.sigmoid(self.conv5(x))
        return x

   
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()





