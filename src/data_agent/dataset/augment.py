#######################################################################################################
# This file is borrowed from COiLTRAiNE https://github.com/felipecode/coiltraine by Felipe Codevilla  #
# COiLTRAiNE itself is under MIT License                                                              #
#######################################################################################################


from imgaug import augmenters as iaa
from torchvision.transforms import Compose, ToTensor, Normalize


def get_image_transform():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def augmenter(frequency_factor=0.5, dropout_factor=0.2, blur_factor=0.5, add_range=10, multi_range=0.2, contrast=0.2):
    return iaa.Sequential([
        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, dropout_factor), per_channel=True)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(0.08, 0.2), per_channel=True)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=True)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Add((-add_range, add_range), per_channel=True)),
        # change brightness of images
        iaa.Sometimes(frequency_factor, iaa.Multiply((1 - multi_range, 1 + multi_range), per_channel=True)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.contrast.LinearContrast((1 - contrast, 1 + contrast), per_channel=True)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale
    ],
        random_order=True  # do all of the above in random order
    )
