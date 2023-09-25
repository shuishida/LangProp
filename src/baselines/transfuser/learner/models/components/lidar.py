import timm
import torch
from torch import nn


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False, in_chans=in_channels)
        self._model.fc = None
        self._model.head = nn.Sequential()

        if architecture.startswith('regnet'):  # Rename modules so we can use the same code
            self._model.conv1 = self._model.stem.conv
            self._model.bn1 = self._model.stem.bn
            self._model.act1 = nn.Sequential()  # The Relu is part of the batch norm here
            self._model.maxpool = nn.Sequential()  # This is used in ResNets
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        elif architecture.startswith('convnext'):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()  # ConvNext does not use an activation function after the stem.
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        return self._model.forward_features(x)
