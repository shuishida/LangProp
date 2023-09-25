import torch
from torch import nn
from torch.nn import functional as F


class ImageDecoder(nn.Module):
    def __init__(self, config, num_class, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_class = num_class

        self.deconv1 = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
            nn.ReLU(True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_3, self.num_class, 3, 1, 1),
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)

        return x


class DepthDecoder(ImageDecoder):
    def __init__(self, config, latent_dim=512):
        super().__init__(config, num_class=1, latent_dim=latent_dim)

    def forward(self, x):
        return torch.sigmoid(super().forward(x)).squeeze(1)
