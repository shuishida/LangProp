from typing import Tuple

import einops
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from baselines.transfuser.learner.models_refactored.backbone import BackboneBase


class GeometricFusionBackbone(BackboneBase):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=False):
        super().__init__(config, image_architecture, lidar_architecture, use_velocity)
        self.use_velocity_final = False

        self.image_convs = nn.ModuleList()
        self.lidar_convs = nn.ModuleList()
        self.image_deconvs = nn.ModuleList()
        self.lidar_deconvs = nn.ModuleList()
        self.image_projections = nn.ModuleList()
        self.lidar_projections = nn.ModuleList()
        self.vel_embs = nn.ModuleList()

        self.hid_dim = hid_dim = config.n_embd

        for i in range(4):
            n_channels = self.image_encoder.feature_info[i]['num_chs']
            self.image_convs.append(nn.Conv2d(n_channels, hid_dim, 1))
            self.lidar_convs.append(nn.Conv2d(n_channels, hid_dim, 1))
            self.image_deconvs.append(nn.Conv2d(hid_dim, n_channels, 1))
            self.lidar_deconvs.append(nn.Conv2d(hid_dim, n_channels, 1))
            self.image_projections.append(
                nn.Sequential(
                    nn.Conv2d(hid_dim, hid_dim, 1), nn.ReLU(),
                    nn.Conv2d(hid_dim, hid_dim, 1), nn.ReLU(),
                    nn.Conv2d(hid_dim, hid_dim, 1), nn.ReLU()
                )
            )
            self.lidar_projections.append(
                nn.Sequential(
                    nn.Conv2d(hid_dim, hid_dim, 1), nn.ReLU(),
                    nn.Conv2d(hid_dim, hid_dim, 1), nn.ReLU(),
                    nn.Conv2d(hid_dim, hid_dim, 1), nn.ReLU()
                )
            )
            self.vel_embs.append(nn.Linear(1, n_channels) if self.use_velocity else nn.Identity())

    def encoder(self, image, lidar, velocity, bev_points, img_points)-> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
            bev_points (tensor): projected image pixels onto the BEV grid. (B H W 5 2)
            cam_points (tensor): projected LiDAR point cloud onto the image space
        '''
        image_features = self.image_encoder.stem(image)
        lidar_features = self.lidar_encoder.stem(lidar)

        for i, (image_block, lidar_block, image_conv, lidar_conv,
                image_deconv, lidar_deconv, image_projection, lidar_projection, vel_emb_layer) in enumerate(zip(
            self.image_encoder.layers, self.lidar_encoder.layers, self.image_convs, self.lidar_convs,
            self.image_deconvs, self.lidar_deconvs, self.image_projections, self.lidar_projections, self.vel_embs
        )):

            image_features = image_block(image_features)
            lidar_features = lidar_block(lidar_features)
            vel_emb = vel_emb_layer(velocity).unsqueeze(-1).unsqueeze(-1)

            scale_factor = max(3 - i, 0)

            def interpolate(x):
                if scale_factor > 0:
                    return F.interpolate(x, scale_factor=2 ** scale_factor, mode='bilinear', align_corners=False)
                return x

            if self.config.n_scale >= scale_factor:
                # fusion at (B, 64, 64, 64)
                image_embed = image_conv(image_features)
                image_embed = self.avgpool_img(image_embed)

                lidar_embed = lidar_conv(lidar_features)
                lidar_embed = self.avgpool_lidar(lidar_embed)

                bev_encoding = rearrange(image_embed, "b c h w -> h w b c")[bev_points[..., 1], bev_points[..., 0]]
                bev_encoding = einops.reduce(bev_encoding, "h w l b c -> b c h w", "sum")
                bev_encoding = image_projection(bev_encoding)

                lidar_extra_features = interpolate(bev_encoding)
                lidar_extra_features = lidar_deconv(lidar_extra_features)
                lidar_features = lidar_features + lidar_extra_features

                if self.use_velocity:
                    lidar_features = lidar_features + vel_emb

                # project bev features to image
                img_encoding = rearrange(lidar_embed, "b c h w -> h w b c")[img_points[..., 1], img_points[..., 0]]
                img_encoding = einops.reduce(img_encoding, "h w l b c -> b c h w", "sum")
                img_encoding = lidar_projection(img_encoding)

                image_extra_features = interpolate(img_encoding)
                image_extra_features = image_deconv(image_extra_features)
                image_features = image_features + image_extra_features

                if self.use_velocity:
                    image_features = image_features + vel_emb

        return image_features, lidar_features
