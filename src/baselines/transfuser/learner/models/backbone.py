import torch
from torch import nn
import torch.nn.functional as F

from baselines.transfuser.learner.models.components.gpt import GPT
from baselines.transfuser.learner.models.components.imagecnn import ImageCNN
from baselines.transfuser.learner.models.components.lidar import LidarEncoder


class BackboneBase(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config
        self.image_architecture = image_architecture
        self.lidar_architecture = lidar_architecture
        self.use_velocity = use_velocity

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)

        in_channels = config.num_features[-1] if config.use_point_pillars else 2 * config.lidar_seq_len
        if self.config.use_target_point_image:
            in_channels += 1
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)

        # Init convert channels
        out_feature_dim = self.image_encoder.features.num_features
        if out_feature_dim == self.config.perception_output_features:
            self.change_channel_conv_image = nn.Identity()
            self.change_channel_conv_lidar = nn.Identity()
        else:
            self.change_channel_conv_image = nn.Conv2d(out_feature_dim, self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(out_feature_dim, self.config.perception_output_features, (1, 1))

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)

        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))
        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return p2

    def forward(self, image, lidar, velocity, bev_points, img_points):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
            bev_points (tensor): projected image pixels onto the BEV grid
            cam_points (tensor): projected LiDAR point cloud onto the image space
        '''
        # return features, image_features_grid, fused_features
        raise NotImplemented

    def _make_gpt(self, in_features):
        config = self.config
        return GPT(n_embd=in_features,
                   n_head=config.n_head,
                   block_exp=config.block_exp,
                   n_layer=config.n_layer,
                   img_vert_anchors=config.img_vert_anchors,
                   img_horz_anchors=config.img_horz_anchors,
                   lidar_vert_anchors=config.lidar_vert_anchors,
                   lidar_horz_anchors=config.lidar_horz_anchors,
                   seq_len=config.seq_len,
                   embd_pdrop=config.embd_pdrop,
                   attn_pdrop=config.attn_pdrop,
                   resid_pdrop=config.resid_pdrop,
                   config=config, use_velocity=self.use_velocity)
