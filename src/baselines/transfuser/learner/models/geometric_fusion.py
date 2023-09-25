import torch
from torch import nn
import torch.nn.functional as F

from baselines.transfuser.learner.models.backbone import BackboneBase
from baselines.transfuser.learner.models.components.utils import normalize_imagenet


class GeometricFusionBackbone(BackboneBase):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=False):
        super().__init__(config, image_architecture, lidar_architecture, use_velocity)

        self.image_conv1 = nn.Conv2d(self.image_encoder.features.feature_info[1]['num_chs'], config.n_embd, 1)
        self.image_conv2 = nn.Conv2d(self.image_encoder.features.feature_info[2]['num_chs'], config.n_embd, 1)
        self.image_conv3 = nn.Conv2d(self.image_encoder.features.feature_info[3]['num_chs'], config.n_embd, 1)
        self.image_conv4 = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], config.n_embd, 1)
        self.image_deconv1 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[1]['num_chs'], 1)
        self.image_deconv2 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[2]['num_chs'], 1)
        self.image_deconv3 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[3]['num_chs'], 1)
        self.image_deconv4 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[4]['num_chs'], 1)

        if self.use_velocity:
            self.vel_emb1 = nn.Linear(1, self.image_encoder.features.feature_info[1]['num_chs'])
            self.vel_emb2 = nn.Linear(1, self.image_encoder.features.feature_info[2]['num_chs'])
            self.vel_emb3 = nn.Linear(1, self.image_encoder.features.feature_info[3]['num_chs'])
            self.vel_emb4 = nn.Linear(1, self.image_encoder.features.feature_info[4]['num_chs'])

        self.lidar_conv1 = nn.Conv2d(self.image_encoder.features.feature_info[1]['num_chs'], config.n_embd, 1)
        self.lidar_conv2 = nn.Conv2d(self.image_encoder.features.feature_info[2]['num_chs'], config.n_embd, 1)
        self.lidar_conv3 = nn.Conv2d(self.image_encoder.features.feature_info[3]['num_chs'], config.n_embd, 1)
        self.lidar_conv4 = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], config.n_embd, 1)
        self.lidar_deconv1 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[1]['num_chs'], 1)
        self.lidar_deconv2 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[2]['num_chs'], 1)
        self.lidar_deconv3 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[3]['num_chs'], 1)
        self.lidar_deconv4 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[4]['num_chs'], 1)

        hid_dim = config.n_embd
        self.image_projection1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection3 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection4 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection3 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection4 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim),
                                               nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))

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

        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        lidar_tensor = lidar

        bz = lidar_tensor.shape[0]

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
        if self.config.n_scale >= 4:
            # fusion at (B, 64, 64, 64)
            image_embd_layer1 = self.image_conv1(image_features)
            image_embd_layer1 = self.avgpool_img(image_embd_layer1)
            lidar_embd_layer1 = self.lidar_conv1(lidar_features)
            lidar_embd_layer1 = self.avgpool_lidar(lidar_embd_layer1)

            curr_h_image, curr_w_image = image_embd_layer1.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer1.shape[-2:]

            # project image features to bev
            bev_points_layer1 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer1 = image_embd_layer1.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer1[:, 1],
                                  bev_points_layer1[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer1 = torch.diagonal(bev_encoding_layer1, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer1 = torch.sum(bev_encoding_layer1, -1)
            bev_encoding_layer1 = self.image_projection1(bev_encoding_layer1.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            lidar_features_layer1 = F.interpolate(bev_encoding_layer1, scale_factor=8, mode='bilinear',
                                                  align_corners=False)
            lidar_features_layer1 = self.lidar_deconv1(lidar_features_layer1)
            lidar_features = lidar_features + lidar_features_layer1
            if self.use_velocity:
                vel_embedding1 = self.vel_emb1(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding1

            # project bev features to image
            img_points_layer1 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer1 = lidar_embd_layer1.permute(0, 2, 3, 1).contiguous()[:, img_points_layer1[:, 1],
                                  img_points_layer1[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer1 = torch.diagonal(img_encoding_layer1, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer1 = torch.sum(img_encoding_layer1, -1)
            img_encoding_layer1 = self.lidar_projection1(img_encoding_layer1.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            image_features_layer1 = F.interpolate(img_encoding_layer1, scale_factor=8, mode='bilinear',
                                                  align_corners=False)
            image_features_layer1 = self.image_deconv1(image_features_layer1)
            image_features = image_features + image_features_layer1

            if self.use_velocity:
                image_features = image_features + vel_embedding1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        if self.config.n_scale >= 3:
            # fusion at (B, 128, 32, 32)
            image_embd_layer2 = self.image_conv2(image_features)
            image_embd_layer2 = self.avgpool_img(image_embd_layer2)
            lidar_embd_layer2 = self.lidar_conv2(lidar_features)
            lidar_embd_layer2 = self.avgpool_lidar(lidar_embd_layer2)

            curr_h_image, curr_w_image = image_embd_layer2.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer2.shape[-2:]

            # project image features to bev
            bev_points_layer2 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer2 = image_embd_layer2.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer2[:, 1],
                                  bev_points_layer2[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer2 = torch.diagonal(bev_encoding_layer2, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer2 = torch.sum(bev_encoding_layer2, -1)
            bev_encoding_layer2 = self.image_projection2(bev_encoding_layer2.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            lidar_features_layer2 = F.interpolate(bev_encoding_layer2, scale_factor=4, mode='bilinear',
                                                  align_corners=False)
            lidar_features_layer2 = self.lidar_deconv2(lidar_features_layer2)
            lidar_features = lidar_features + lidar_features_layer2

            if self.use_velocity:
                vel_embedding2 = self.vel_emb2(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding2

            # project bev features to image
            img_points_layer2 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer2 = lidar_embd_layer2.permute(0, 2, 3, 1).contiguous()[:, img_points_layer2[:, 1],
                                  img_points_layer2[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer2 = torch.diagonal(img_encoding_layer2, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer2 = torch.sum(img_encoding_layer2, -1)
            img_encoding_layer2 = self.lidar_projection2(img_encoding_layer2.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            image_features_layer2 = F.interpolate(img_encoding_layer2, scale_factor=4, mode='bilinear',
                                                  align_corners=False)
            image_features_layer2 = self.image_deconv2(image_features_layer2)
            image_features = image_features + image_features_layer2

            if self.use_velocity:
                image_features = image_features + vel_embedding2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        if self.config.n_scale >= 2:
            # fusion at (B, 256, 16, 16)
            image_embd_layer3 = self.image_conv3(image_features)
            image_embd_layer3 = self.avgpool_img(image_embd_layer3)
            lidar_embd_layer3 = self.lidar_conv3(lidar_features)
            lidar_embd_layer3 = self.avgpool_lidar(lidar_embd_layer3)

            curr_h_image, curr_w_image = image_embd_layer3.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer3.shape[-2:]

            # project image features to bev
            bev_points_layer3 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer3 = image_embd_layer3.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer3[:, 1],
                                  bev_points_layer3[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer3 = torch.diagonal(bev_encoding_layer3, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer3 = torch.sum(bev_encoding_layer3, -1)
            bev_encoding_layer3 = self.image_projection3(bev_encoding_layer3.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            lidar_features_layer3 = F.interpolate(bev_encoding_layer3, scale_factor=2, mode='bilinear',
                                                  align_corners=False)
            lidar_features_layer3 = self.lidar_deconv3(lidar_features_layer3)
            lidar_features = lidar_features + lidar_features_layer3

            if self.use_velocity:
                vel_embedding3 = self.vel_emb3(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding3

            # project bev features to image
            img_points_layer3 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer3 = lidar_embd_layer3.permute(0, 2, 3, 1).contiguous()[:, img_points_layer3[:, 1],
                                  img_points_layer3[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer3 = torch.diagonal(img_encoding_layer3, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer3 = torch.sum(img_encoding_layer3, -1)
            img_encoding_layer3 = self.lidar_projection3(img_encoding_layer3.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            image_features_layer3 = F.interpolate(img_encoding_layer3, scale_factor=2, mode='bilinear',
                                                  align_corners=False)
            image_features_layer3 = self.image_deconv3(image_features_layer3)
            image_features = image_features + image_features_layer3
            if self.use_velocity:
                image_features = image_features + vel_embedding3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # fusion at (B, 512, 8, 8)
        if self.config.n_scale >= 1:
            # fusion at (B, 512, 8, 8)
            image_embd_layer4 = self.image_conv4(image_features)
            image_embd_layer4 = self.avgpool_img(image_embd_layer4)
            lidar_embd_layer4 = self.lidar_conv4(lidar_features)
            lidar_embd_layer4 = self.avgpool_lidar(lidar_embd_layer4)

            curr_h_image, curr_w_image = image_embd_layer4.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer4.shape[-2:]

            # project image features to bev
            bev_points_layer4 = bev_points.view(bz * curr_h_lidar * curr_w_lidar * 5, 2)
            bev_encoding_layer4 = image_embd_layer4.permute(0, 2, 3, 1).contiguous()[:, bev_points_layer4[:, 1],
                                  bev_points_layer4[:, 0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer4 = torch.diagonal(bev_encoding_layer4, 0).permute(4, 3, 0, 1, 2).contiguous()
            bev_encoding_layer4 = torch.sum(bev_encoding_layer4, -1)
            bev_encoding_layer4 = self.image_projection4(bev_encoding_layer4.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            lidar_features_layer4 = self.lidar_deconv4(bev_encoding_layer4)
            lidar_features = lidar_features + lidar_features_layer4

            if self.use_velocity:
                vel_embedding4 = self.vel_emb4(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding4

            # project bev features to image
            img_points_layer4 = img_points.view(bz * curr_h_image * curr_w_image * 5, 2)
            img_encoding_layer4 = lidar_embd_layer3.permute(0, 2, 3, 1).contiguous()[:, img_points_layer4[:, 1],
                                  img_points_layer4[:, 0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer4 = torch.diagonal(img_encoding_layer4, 0).permute(4, 3, 0, 1, 2).contiguous()
            img_encoding_layer4 = torch.sum(img_encoding_layer4, -1)
            img_encoding_layer4 = self.lidar_projection4(img_encoding_layer4.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                                                                          2).contiguous()
            image_features_layer4 = self.image_deconv4(img_encoding_layer4)
            image_features = image_features + image_features_layer4
            if self.use_velocity:
                image_features = image_features + vel_embedding4

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

