from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from baselines.transfuser.learner.models_refactored.backbone import BackboneBase


class TransfuserBackbone(BackboneBase):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__(config, image_architecture, lidar_architecture, use_velocity)

        self.transformers = nn.ModuleList()
        for i in range(4):
            n_channels = self.image_encoder.feature_info[i]['num_chs']
            self.transformers.append(self._make_gpt(n_channels))

    def encoder(self, image, lidar, velocity, bev_points, img_points)-> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image: input rgb image
            lidar: LiDAR input will be replaced by positional encoding. Third channel may contain target point.
            velocity (tensor): input velocity from speedometer
        '''
        image_features = self.image_encoder.stem(image)
        lidar_features = self.lidar_encoder.stem(lidar)

        for i, (image_block, lidar_block, transformer) in enumerate(zip(
                self.image_encoder.layers, self.lidar_encoder.layers, self.transformers
        )):
            image_features = image_block(image_features)
            lidar_features = lidar_block(lidar_features)

            image_embed = self.avgpool_img(image_features)
            lidar_embed = self.avgpool_lidar(lidar_features)

            image_extra_features, lidar_extra_features = transformer(image_embed, lidar_embed, velocity)

            image_extra_features = F.interpolate(image_extra_features,
                                                 size=image_features.shape[-2:], mode='bilinear', align_corners=False)
            lidar_extra_features = F.interpolate(lidar_extra_features,
                                                 size=lidar_features.shape[-2:], mode='bilinear', align_corners=False)
            image_features = image_features + image_extra_features
            lidar_features = lidar_features + lidar_extra_features

        return image_features, lidar_features
