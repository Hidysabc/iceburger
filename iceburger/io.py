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


def image_normalization(x, percentile=1, pad_channel="zeros"):
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
    else:
        pad = np.zeros(x.shape[:2] + (1,))

    return np.concatenate([x, pad], axis=-1)[np.newaxis, :, :, :]


def parse_json_data(json_filename, percentile=1, padding="zeros"):
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
    X = np.concatenate([image_normalization(x, percentile, padding)
                        for x in _X], axis=0)
    X_angle = df.inc_angle.values
    y = df.is_iceberg.values
    if "set" in df.columns:
        subset = df["set"].values
    else:
        subset = np.array(["train"] * len(y))

    return (X, X_angle, y, subset)

def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(x, weight=weight, multichannel=multichannel) for x in X])

def smooth(X, sigma):
    return np.asarray([gaussian(x, sigma=sigma) for x in X])

def grayscale(X):
    return np.asarray([rgb2gray(x) for x in X])

# Translate data to an image format
def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)



