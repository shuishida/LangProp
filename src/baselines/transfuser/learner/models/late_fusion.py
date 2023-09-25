import torch
from torch import nn
import timm

from baselines.transfuser.learner.models.backbone import BackboneBase
from baselines.transfuser.learner.models.components.utils import normalize_imagenet


class LateFusionBackbone(BackboneBase):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=False):
        super().__init__(config, image_architecture, lidar_architecture, use_velocity)

        if image_architecture.startswith('convnext'):
            self.norm_after_pool_img = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_img = nn.Identity()

        if lidar_architecture.startswith('convnext'):
            self.norm_after_pool_lidar = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_lidar = nn.Identity()

        # velocity embedding
        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb = nn.Linear(1, self.config.perception_output_features)

    def forward(self, image, lidar, velocity, bev_points, img_points):
        '''
        Image + LiDAR feature fusion
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        # Image branch
        output_features_image = self.image_encoder(image_tensor)
        output_features_image = self.reduce_channels_conv_image(output_features_image)
        image_features_grid = output_features_image

        image_features = torch.nn.AdaptiveAvgPool2d((1, 1))(output_features_image)
        image_features = torch.flatten(image_features, 1)
        image_features = self.norm_after_pool_img(image_features)

        # LiDAR branch
        output_features_lidar = self.lidar_encoder(lidar)
        output_features_lidar = self.reduce_channels_conv_lidar(output_features_lidar)
        lidar_features_grid = output_features_lidar
        features = self.top_down(lidar_features_grid)

        lidar_features = torch.nn.AdaptiveAvgPool2d((1, 1))(output_features_lidar)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = self.norm_after_pool_lidar(lidar_features)

        # Fusion
        fused_features = image_features + lidar_features

        if self.use_velocity:
            velocity_embeddings = self.vel_emb(velocity)  # (B, C) .unsqueeze(1)
            fused_features = fused_features + velocity_embeddings

        return features, image_features_grid, fused_features
