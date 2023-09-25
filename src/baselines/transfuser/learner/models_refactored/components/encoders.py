import timm
from torch import nn


class ImageEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        in_channels: input channels
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, in_channels=3, normalize=True, pretrained=True):
        super().__init__()
        self.architecture = architecture
        self.normalize = normalize
        self._model = timm.create_model(architecture, pretrained=pretrained, in_chans=in_channels)
        self._model.fc = None
        self._model.head = nn.Sequential()
        self.num_features = self._model.num_features

        # Rename modules so we can use the same code
        if self.architecture.startswith('regnet'):
            self.layers = nn.ModuleList([self._model.s1, self._model.s2, self._model.s3, self._model.s4])
            self.feature_info = self._model.feature_info[1:5]

        elif self.architecture.startswith('convnext'):
            self.layers = nn.ModuleList([self._model.stages._modules[str(i)] for i in range(4)])
            # ConvNext don't have the 0th entry that res nets use.
            self.feature_info = self._model.feature_info[:4]

        elif self.architecture.startswith('resnet'):
            self.layers = nn.ModuleList([self._model.layer1, self._model.layer2, self._model.layer3, self._model.layer4])
            # ConvNext don't have the 0th entry that res nets use.
            self.feature_info = self._model.feature_info[1:5]
        else:
            raise NotImplementedError

    def stem(self, x):
        if self.architecture.startswith('resnet'):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)
            return x
        return self._model.stem(x)

    def forward(self, x):
        return self._model.forward_features(x)


class LidarEncoder(ImageEncoder):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2, normalize=True, pretrained=True):
        super().__init__(architecture, in_channels, normalize, pretrained)


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x
