import cv2
import os

def crop_center(image):
    height = image.shape[0]
    width = image.shape[1]
    edge = (width - height) // 2
    return image[0:(height - 1), edge:(edge + height), :]
