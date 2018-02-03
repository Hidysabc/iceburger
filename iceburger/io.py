"""
Functions related to file IO and data preprocessing
"""
import numpy as np
import pandas as pd

# An image clearing dependencies
# from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                 denoise_nl_means)
# from skimage.filters import (gaussian, laplace, sobel, scharr, prewitt, roberts,
#                             frangi, hessian, gabor)
from skimage.color import rgb2gray

from .filters import (smooth_and_denoise, sobel_filter, scharr_filter,
                      prewitt_filter, roberts_filter, frangi_filter,
                      hessian_filter, laplace_filter, gabor_filter,
                      lee_filter, frost_filter)
"""
def denoise(X, weight, multichannel):
    return denoise_tv_chambolle(X, weight=weight, multichannel=multichannel)


def smooth(X, sigma):
    return gaussian(X, sigma=sigma)

def sobel_filter(X):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(sobel(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]


def scharr_filter(X):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(scharr(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def prewitt_filter(X):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(prewitt(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def roberts_filter(X):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(roberts(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def frangi_filter(X, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(frangi(X, scale_range=scale_range, scale_step=scale_step, beta1=beta1, beta2=beta2)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def hessian_filter(X, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(hessian(X, scale_range=scale_range, scale_step=scale_step, beta1=beta1, beta2=beta2)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def laplace_filter(X, kernel=3):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(laplace(band, ksize=kernel)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def gabor_filter(X, frequency, theta=0, bandwidth=1,
                 sigma_x=None, sigma_y=None, n_stds=3, offset=0,
                 mode="reflect", cval=0):
    arr_r=  []
    arr_i = []
    for i in range(X.shape[2]):
        r, i = gabor(band, frequency=frequency, theta=theta, bandwidth=bandwidth,
                     sigma_x=sigma_x, sigma_y=sigma_y, nstds=nstds, offset=offset,
                     mode=mode, cval=cval)
        arr_r.append(r[:,:,np.newaxis])
        arr_i.append(i[:,:,np.newaxis])
    return np.concatenate(arr_r,axis=2)[np.newaxis,:,:,:], np.concatenate(arr_i, axis=2)[np.newaxis,:,:,:]
"""
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
    elif pad_channel == "minus":
        pad = ((x[:, :, 0] - x[:, :, 1]))[:, :, np.newaxis]
    elif pad_channel == "zeros":
        pad = np.zeros(x.shape[:2] + (1,))
    else:
        raise ValueError("Unrecognize pad_channel option: {}"
                         .format(pad_channel))

    return np.concatenate([x, pad], axis=-1)[np.newaxis, :, :, :]
'''
def map_uint8(x):
    """Map the band value to unsigned int 8 in order to try the imagaug

    :param x: numpy.ndarray of signal of dimension (height, weight)
    :returns x: the mapped unsigned int format of x which divides by 255 for fair comparison with the original
    """
    vmin = np.min(x)
    vmax = np.max(x)
    if (vmin ==0) and (vmax==0):
        x = np.zeros(x.shape)
    else:
        x = (x - vmin) * 255  / (vmax - vmin)
    x = np.uint8(x) / 255
    return x

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
        smoothed = smooth(denoise(band, weight_gray, False),
                          smooth_gray)
        # smoothed = map_uint8(smoothed)[:, :, np.newaxis]
        smoothed = smoothed[:, :, np.newaxis]
        arr.append(smoothed)
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]
'''

def apply_filter(X, filter_instance, **filter_args):
    return np.concatenate([filter_instance(x, **filter_args) for x in X], axis=0)



def parse_json_data(json_filename, percentile=1, padding="zeros", smooth=True,
                    filter_instance=None, **filter_args):
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
        # X_norm = np.concatenate([image_normalization(x, percentile, padding)
        #                         for x in _X], axis=0)
        X = np.concatenate([smooth_and_denoise(x, padding)
                            for x in _X], axis=0)

    if filter_instance is not None:
        # X = apply_filter(X, lee_filter)
        X1 = apply_filter(X, sobel_filter)
        X2 = apply_filter(X, filter_instance, **filter_args)
        # X3 = apply_filter(X, prewitt_filter)
        X = np.concatenate([X1, X2], axis = -1)

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
