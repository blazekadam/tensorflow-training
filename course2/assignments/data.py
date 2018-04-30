from typing import Tuple, Iterable

import pandas as pd
import numpy as np


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read facial key-points data from the given file, filter NAs and return (images, targets) pair.

    Include only 'left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                 'nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y' and 'Image'
                 columns.

    Return shapes and dtypes shoould be:
        - images: (7000, 96, 96, 1), np.uint8
        - targets: (7000, 4, 2), np.float32

    :param filename:
    :return:
    """


def batch_data(images: np.ndarray, targets: np.ndarray, batch_size: int=100) \
        -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Shuffle and iterate through the given image and target data creating batches of the specified size.

    :param images: images to be iterated (None, 96, 96, 1)
    :param target: targets to be iterated (None, 4, 2)
    :param batch_size: batch size
    :return: an iterable of (images, targets) tuples (batches of the specified size)
    """
