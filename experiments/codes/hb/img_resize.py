import cv2
import numpy as np


def bulk_resize(img_data, new_size):
    """
    img_data: np-array containing all images
    new_size: size of the new images to produce
    """
    resized_img_data = np.array([cv2.resize(item, new_size) for item in img_data])
    return resized_img_data
