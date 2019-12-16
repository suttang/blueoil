import random

import numpy as np
import PIL.Image


def load_image(filename, convert_rgb=True):
    """Returns numpy array of an image"""
    image = PIL.Image.open(filename)

    #  sometime image data is gray.
    if convert_rgb:
        image = image.convert("RGB")
    else:
        image = image.convert("L")

    image = np.array(image)

    return image


def crop(image, patch_size):
    height, width = image.shape[0:2]

    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)

    if np.ndim(image) == 2:
        return image[top:top + patch_size, left:left + patch_size]

    return image[top:top + patch_size, left:left + patch_size, :]


def scale(image, scale, method=PIL.Image.BICUBIC):
    height, width = image.shape[0:2]

    new_width = int(width * scale)
    new_height = int(height * scale)

    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB
        image = PIL.Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA
        image = PIL.Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = PIL.Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)

    return image


def convert_rgb_to_ycbcr(image):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    xform = np.array(
        [
            [65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
            [-37.945 / 256.0, -74.494 / 256.0, 112.439 / 256.0],
            [112.439 / 256.0, -94.154 / 256.0, -18.285 / 256.0],
        ]
    )

    ycbcr_image = image.dot(xform.T)
    ycbcr_image[:, :, 0] += 16.0
    ycbcr_image[:, :, [1, 2]] += 128.0

    return ycbcr_image


def convert_ycbcr_to_rgb(ycbcr_image):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])

    rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
    rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
    xform = np.array(
        [
            [298.082 / 256.0, 0, 408.583 / 256.0],
            [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
            [298.082 / 256.0, 516.412 / 256.0, 0],
        ]
    )
    rgb_image = rgb_image.dot(xform.T)

    return rgb_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image):
    if len(y_image.shape) <= 2:
        y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image[:, :, 0]
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image)
