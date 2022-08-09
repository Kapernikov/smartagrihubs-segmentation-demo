import os

import numpy as np
import cv2 as cv

def write_image_cv(img: np.array, img_name: str) -> None:
    """ basic opencv image writer """
    FOLDER = 'tmp_images'
    os.makedirs(FOLDER, exist_ok=True)

    if img.max() <= 1:
        img = img*255

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(os.path.join(FOLDER, img_name), img)

def write_dataset(X: np.array, y: np.array) -> None:
    """ write all images and masks in the dataset """
    for i in range(X.shape[0]):
        write_image_cv(X[i,:,:,:], f'img_{i}.png')
        write_image_cv(y[i,:,:,0], f'msk_{i}.png') 