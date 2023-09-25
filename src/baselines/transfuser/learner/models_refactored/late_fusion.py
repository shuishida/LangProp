from typing import Tuple

import torch

from baselines.transfuser.learner.models_refactored.backbone import BackboneBase


class LateFusionBackbone(BackboneBase):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """
    def encoder(self, image, lidar, velocity, bev_points, img_points) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image_encoder(image), self.lidar_encoder(lidar)
