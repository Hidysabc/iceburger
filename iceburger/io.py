"""
Functions related to file IO and data preprocessing
"""
import numpy as np
import pandas as pd

# An image clearing dependencies
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_nl_means)
from skimage.filters import gaussian
from skimage.color import rgb2gray


def denoise(X, weight, multichannel):
    return denoise_tv_chambolle(X, weight=weight, multichannel=multichannel)


def smooth(X, sigma):
    return gaussian(X, sigma=sigma)


def grayscale(X):
    return np.asarray([rgb2gray(x) for x in X])


# Translate data to an image format
def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2
        # band_3 = (band_1 + band_2) / 2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)


def image_normalization(x, percentile, pad_channel):
    """Normalize the image signal value by rescale data

    :param x: :class:`numpy.ndarray` of signal of dimension (height, width, 2)
    :param percentile: signal greater or less than the percentile will be capped
        as 1 and 0 respectively
    :returns: :class:`numpy.ndarray` of normalized 3 channel image with last
        channel totally black
    """
    vmax = np.percentile(x, 100 - percentile)
    vmin = np.percentile(x, percentile)
    x = (x - vmin) / (vmax - vmin)
    x[x > 1] = 1
    x[x < 0] = 0
    if pad_channel == "avg":
        pad = ((x[:, :, 0] + x[:, :, 1]) / 2)[:, :, np.newaxis]
    elif pad_channel == "zeros":
        pad = np.zeros(x.shape[:2] + (1,))
    else:
        raise ValueError("Unrecognize pad_channel option: {}"
                         .format(pad_channel))

    return np.concatenate([x, pad], axis=-1)[np.newaxis, :, :, :]


def smooth_and_denoise(x, padding, smooth_gray=0.2, weight_rgb=0.1,
                       weight_gray=0.1):
    if padding == "avg":
        pad = ((x[:, :, 0] + x[:, :, 1]) / 2)[:, :, np.newaxis]
    elif padding == "zeros":
        pad = np.zeros(x.shape[:2] + (1,))
    else:
        raise ValueError("Unrecognize pad_channel option: {}"
                         .format(padding))

    x = np.concatenate([x, pad], axis=-1)
    arr = []
    for i in range(x.shape[2]):
        band = x[:, :, i]
        arr.append(smooth(denoise(band, weight_gray, False), smooth_gray)
                   [:, :, np.newaxis])

    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]


def parse_json_data(json_filename, percentile=1, padding="zeros", smooth=True):
    """Parse json data to generate trainable matrices

    :param json_filename: path to input json file
    :returns: a `tuple` of
        X: :class:`numpy.ndarray` of dimension (nb_samples, height, width, 3)
        X_angle: :class:`numpy.array` of dimension (nb_samples) of incidence
            angles
        y: :class:`numpy.array` of labels
    """
    df = pd.read_json(json_filename)
    dim = int(np.sqrt(len(df.band_1.iloc[0])))
    _X = np.concatenate([
        np.concatenate([np.array(r.band_1).reshape((dim, dim, 1)),
                        np.array(r.band_2).reshape((dim, dim, 1))],
                       axis=-1)[np.newaxis, :, :, :]
        for _, r in df.iterrows()], axis=0)
    if not smooth:
        X = np.concatenate([image_normalization(x, percentile, padding)
                            for x in _X], axis=0)
    else:
        X = np.concatenate([smooth_and_denoise(x, padding)
                            for x in _X], axis=0)

    X_angle = df.inc_angle.values
    if "is_iceberg" in df.columns:
        y = df.is_iceberg.values
    else:
        y = None

    if "set" in df.columns:
        subset = df["set"].values
    else:
        subset = np.array(["train"] * X.shape[0])

    ID = df.id
    return (ID, X, X_angle, y, subset)
