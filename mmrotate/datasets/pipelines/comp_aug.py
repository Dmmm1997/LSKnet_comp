# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Union

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RandomBlur(BaseTransform):
    """Convert boxes in results to a certain box type.

    Args:
        box_type_mapping (dict): A dictionary whose key will be used to search
            the item in `results`, the value is the destination box type.
    """

    def __init__(self, prob: float, value_range: list = [5, 19]) -> None:
        self.prob = prob
        self.value_range = value_range
        self.blur_type = [0, 1, 2]

    @cache_randomness
    def _random_blur_type(self) -> int:
        """Random blur type."""
        return np.random.choice(self.blur_type, 1)[0]

    @cache_randomness
    def _is_blur(self) -> bool:
        """Randomly decide whether to blur."""
        return np.random.rand() < self.prob

    @cache_randomness
    def _random_blur_value(self) -> int:
        """Random blur value."""
        return np.random.choice(
            list(range(self.value_range[0], self.value_range[1], 1)), 1)[0]

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_blur():
            return results

        type = self._random_blur_type()
        value = self._random_blur_value()
        img = results['img']
        value = value - 1 if value % 2 == 0 else value
        if type == 0:
            img = cv2.blur(img, (value, value))
        elif type == 1:
            img = cv2.medianBlur(img, value)
        elif type == 2:
            img = cv2.GaussianBlur(img, (value, value), 0)
        else:
            raise TypeError("Blur type is not existed!!!")
        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(blur_type={self.blur_type})'
        repr_str += f'(blur_prob={self.prob})'
        repr_str += f'(blur_value_range={self.value_range})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class RandomNoise(BaseTransform):
    """Convert boxes in results to a certain box type.

    Args:
        box_type_mapping (dict): A dictionary whose key will be used to search
            the item in `results`, the value is the destination box type.
    """

    def __init__(self, prob: float, sigma_range: list = [3, 50]) -> None:
        self.prob = prob
        self.sigma_range = sigma_range

    @cache_randomness
    def _is_noise(self) -> bool:
        """Randomly decide whether to noise."""
        return np.random.rand() < self.prob

    @cache_randomness
    def _random_sigma_value(self) -> int:
        """Random noise value."""
        return np.random.choice(
            list(range(self.sigma_range[0], self.sigma_range[1], 1)), 1)[0]

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_noise():
            return results

        mean = 0
        sigma = self._random_sigma_value()
        img = results['img']
        img_height, img_width, img_channels = img.shape
        gauss = np.random.normal(mean, sigma,
                                 (img_height, img_width, img_channels))
        #给图片添加高斯噪声
        noisy_img = img + gauss
        #设置图片添加高斯噪声之后的像素值的范围
        noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
        results['img'] = noisy_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(noise_sigma_range={self.sigma_range})'
        repr_str += f'(noise_prob={self.prob})'
        return repr_str
    

@ROTATED_PIPELINES.register_module()
class RandomBrightness(BaseTransform):
    """Convert boxes in results to a certain box type.

    Args:
        box_type_mapping (dict): A dictionary whose key will be used to search
            the item in `results`, the value is the destination box type.
    """

    def __init__(self, prob: float, gamma_range: list = [0.05, 1.5]) -> None:
        self.prob = prob
        self.gamma_range = gamma_range

    @cache_randomness
    def _is_brightness(self) -> bool:
        """Randomly decide whether to brightness."""
        return np.random.rand() < self.prob

    @cache_randomness
    def _random_gamma_value(self) -> int:
        """Random brightness gamma_value."""
        return np.random.choice(
            np.arange(self.gamma_range[0], self.gamma_range[1], 0.01), 1)[0]

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_brightness():
            return results

        gamma = self._random_gamma_value()
        img = results['img']
        brightness_image = np.power(img/255.0, gamma)
        brightness_image = np.uint8(brightness_image * 255)
        brightness_image = np.clip(brightness_image, a_min=0, a_max=255)
        results['img'] = brightness_image
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_gamma_range={self.gamma_range})'
        repr_str += f'(brightness_prob={self.prob})'
        return repr_str