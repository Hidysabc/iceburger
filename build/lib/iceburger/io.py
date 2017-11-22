"""
Functions related to file IO and data preprocessing
"""

import numpy as np
import pandas as pd


def image_normalization(x, percentile=1):
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
    return np.concatenate([x, np.zeros(x.shape[:2] + (1,))],
                          axis=-1)[np.newaxis, :, :, :]


def parse_json_data(json_filename):
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
    X = np.concatenate([image_normalization(x) for x in _X], axis=0)
    X_angle = df.inc_angle.values
    y = df.is_iceberg.values
    if "set" in df.columns:
        subset = df["set"].values
    else:
        subset = np.array(["train"] * len(y))

    return (X, X_angle, y, subset)