import torch
from torch import nn
import torch.nn.functional as F

from baselines.transfuser.learner.models.backbone import BackboneBase
from baselines.transfuser.learner.models.components.utils import normalize_imagenet


class TransfuserBackbone(BackboneBase):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__(config, image_architecture, lidar_architecture, use_velocity)

        self.transformer1 = self._make_gpt(self.image_encoder.features.feature_info[1]['num_chs'])
        self.transformer2 = self._make_gpt(self.image_encoder.features.feature_info[2]['num_chs'])
        self.transformer3 = self._make_gpt(self.image_encoder.features.feature_info[3]['num_chs'])
        self.transformer4 = self._make_gpt(self.image_encoder.features.feature_info[4]['num_chs'])

    def forward(self, image, lidar, velocity, bev_points, img_points):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''

        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        lidar_tensor = lidar

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        # Image fusion at (B, 72, 40, 176)
        # Lidar fusion at (B, 72, 64, 64)
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)

        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(image_features_layer1,
                                              size=(image_features.shape[2], image_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1,
                                              size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        # Image fusion at (B, 216, 20, 88)
        # Image fusion at (B, 216, 32, 32)
        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(image_features_layer2,
                                              size=(image_features.shape[2], image_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2,
                                              size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        # Image fusion at (B, 576, 10, 44)
        # Image fusion at (B, 576, 16, 16)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(image_features_layer3,
                                              size=(image_features.shape[2], image_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3,
                                              size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # Image fusion at (B, 1512, 5, 22)
        # Image fusion at (B, 1512, 8, 8)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)

        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features_layer4 = F.interpolate(image_features_layer4,
                                              size=(image_features.shape[2], image_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        lidar_features_layer4 = F.interpolate(lidar_features_layer4,
                                              size=(lidar_features.shape[2], lidar_features.shape[3]), mode='bilinear',
                                              align_corners=False)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

        # Downsamples channels to 512
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        x4 = lidar_features
        image_features_grid = image_features  # For auxilliary information

        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        fused_features = image_features + lidar_features

        features = self.top_down(x4)
        return features, image_features_grid, fused_features
