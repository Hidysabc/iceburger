"""
Functions related to file IO and data preprocessing
"""
import numpy as np
import pandas as pd

# An image clearing dependencies
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_nl_means)

from skimage.filters import (gaussian, laplace, sobel, scharr, prewitt, roberts,
                             frangi, hessian, gabor)
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.stats import variation

def denoise(X, weight, multichannel):
    return denoise_tv_chambolle(X, weight=weight, multichannel=multichannel)


def smooth(X, sigma):
    return gaussian(X, sigma=sigma)

def sobel_filter(X):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(sobel(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]


def scharr_filter(X):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(scharr(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def prewitt_filter(X):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(prewitt(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def roberts_filter(X):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(roberts(band)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def frangi_filter(X, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(frangi(X, scale_range=scale_range, scale_step=scale_step, beta1=beta1, beta2=beta2)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def hessian_filter(X, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(hessian(X, scale_range=scale_range, scale_step=scale_step, beta1=beta1, beta2=beta2)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def laplace_filter(X, kernel=3):
    arr = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        arr.append(laplace(band, ksize=kernel)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

def gabor_filter(X, frequency, theta=0, bandwidth=1,
                 sigma_x=None, sigma_y=None, n_stds=3, offset=0,
                 mode="reflect", cval=0):
    arr_r0=  []
    arr_i0 = []
    arr_r1 = []
    arr_i1 = []
    arr_r2 = []
    arr_i2 = []
    # arr_r3 = []
    # arr_i3 = []
    for i in range(X.shape[2]-1):
        band = X[:, :, i]
        r0, i0 = gabor(band, frequency=frequency, theta=np.pi*0/180, bandwidth=bandwidth,
                     sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=offset,
                     mode=mode, cval=cval)
        r1, i1= gabor(band, frequency=frequency, theta=np.pi*45/180, bandwidth=bandwidth,
                     sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=offset,
                     mode=mode, cval=cval)
        r2, i2= gabor(band, frequency=frequency, theta=np.pi*90/180, bandwidth=bandwidth,
                     sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=offset,
                     mode=mode, cval=cval)
        # r3, i3= gabor(band, frequency=frequency, theta=np.pi*15/180, bandwidth=bandwidth,
        #             sigma_x=sigma_x, sigma_y=sigma_y, n_stds=n_stds, offset=offset,
        #             mode=mode, cval=cval)
        arr_r0.append(r0[:,:,np.newaxis])
        arr_i0.append(i0[:,:,np.newaxis])
        arr_r1.append(r1[:,:,np.newaxis])
        arr_i1.append(i1[:,:,np.newaxis])
        arr_r2.append(r2[:,:,np.newaxis])
        arr_i2.append(i2[:,:,np.newaxis])
        # arr_r3.append(r2[:,:,np.newaxis])
        # arr_i3.append(i2[:,:,np.newaxis])
    return np.concatenate(np.concatenate([arr_r0, arr_i0, arr_r1, arr_i1, arr_r2, arr_i2], axis=-1), axis=2)[np.newaxis,:,:,:]

def lee_filter(X, size=3, noise_factor=1):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        band_mean = uniform_filter(band, (size, size))
        band_sqr_mean = uniform_filter(band*band, (size, size))
        band_variance = band_sqr_mean - band_mean*band_mean
        overall_variance = variance(band)
        band_weights = band_variance * band_variance / (band_variance * band_variance\
                                                + (overall_variance*noise_factor)**2)
        band_output = band_mean + band_weights * (band - band_mean)
        arr.append(band_output[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]



COEF_VAR_DEFAULT = 0.01

def compute_coef_var(image, x_start, x_end, y_start, y_end):
    """
    Compute coefficient of variation in a window of [x_start: x_end] and
    [y_start:y_end] within the image.
    """
    # assert x_start >= 0, 'ERROR: x_start must be >= 0.'
    # assert y_start >= 0, 'ERROR: y_start must be >= 0.'

    x_size, y_size = image.shape
    x_overflow = x_end > x_size
    y_overflow = y_end > y_size

    # assert not x_overflow, 'ERROR: invalid parameters cause x window overflow.'
    # assert not y_overflow, 'ERROR: invalid parameters cause y window overflow.'

    window = image[x_start:x_end, y_start:y_end]

    coef_var = variation(window, None)

    if not coef_var or np.isnan(coef_var) or coef_var==0:  # dirty patch
        coef_var = COEF_VAR_DEFAULT
        # print "squared_coef was equal zero but replaced by %s" % coef_var
    # assert coef_var > 0, 'ERROR: coeffient of variation cannot be zero.'

    return coef_var


def calculate_all_Mi(window_flat, factor_A, window):
    """
    Compute all the weights of pixels in the window.
    """
    N, M = window.shape
    center_pixel = np.float64(window[ N // 2, M // 2])
    window_flat = np.float64(window_flat)

    distances = np.abs(window_flat - center_pixel)

    weights = np.exp(-factor_A * distances)

    return weights


def calculate_local_weight_matrix(window, factor_A):
    """
    Returns an array with the weights for the pixels in the given window.
    """
    weights_array = np.zeros(window.size)
    window_flat = window.flatten()

    weights_array = calculate_all_Mi(window_flat, factor_A, window)

    return weights_array


def frost_filter_(img, damping_factor=2.0, win_size=3):
    """
    Apply frost filter to a numpy matrix containing the image, with a window of
    win_size x win_size.
    By default, the window size is 3x3.
    """

    # assert_window_size(win_size)

    img_filtered = np.zeros_like(img)
    N, M = img.shape
    win_offset = win_size / 2
    mean_offset = 1e-17
    for i in range(0, N):
        xleft = int(i - win_offset)
        xright = int(i + win_offset)
        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N - 1
        for j in range(0, M):
            yup = int(j - win_offset)
            ydown = int(j + win_offset)
            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M - 1

            # assert_indices_in_range(N, M, xleft, xright, yup, ydown)

            # inspired by http://www.pcigeomatics.com/cgi-bin/pcihlp/FFROST
            variation_coef = compute_coef_var(img, xleft, xright, yup, ydown)
            window = img[xleft:xright, yup:ydown]
            window_mean = window.mean()
            sigma_zero = variation_coef / (window_mean+mean_offset)  # var / u^2
            factor_A = damping_factor * sigma_zero

            weights_array = calculate_local_weight_matrix(window, factor_A)
            pixels_array = window.flatten()

            weighted_values = weights_array * pixels_array
            img_filtered[i, j] = weighted_values.sum() / weights_array.sum()

    return img_filtered

def frost_filter(X, damping_factor=2.0, win_size=3):
    arr = []
    for i in range(X.shape[2]):
        band = X[:, :, i]
        arr.append(frost_filter_(band, damping_factor=damping_factor, \
                                  win_size=win_size)[:, :, np.newaxis])
    return np.concatenate(arr, axis=2)[np.newaxis, :, :, :]

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
    elif padding == "minus":
        pad = ((x[:, :, 0] - x[:, :, 1]))[:, :, np.newaxis]
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


