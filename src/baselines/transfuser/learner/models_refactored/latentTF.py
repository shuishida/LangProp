import torch
from baselines.transfuser.learner.models_refactored.transfuser import TransfuserBackbone


class latentTFBackbone(TransfuserBackbone):
    """
    Multi-scale Fusion Transformer for image + pos_embedding feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """
    def process_lidar(self, lidar):
        x = torch.linspace(-1, 1, self.config.lidar_resolution_width)
        y = torch.linspace(-1, 1, self.config.lidar_resolution_height)
        y_grid, x_grid = torch.meshgrid(x, y, indexing='ij')

        lidar[:, 0] = y_grid.unsqueeze(0)  # Top down positional encoding
        lidar[:, 1] = x_grid.unsqueeze(0)  # Left right positional encoding
        return lidar
