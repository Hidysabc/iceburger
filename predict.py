from __future__ import print_function
from __future__ import division
import numpy as np
import os
import logging
import argparse
import pandas as pd
import sys
#import shutil

from keras import backend as K
#from keras.engine.topology import get_source_inputs
#from keras.layers import (Input, Activation, BatchNormalization, Conv2D,
#                                                    Dense, Dropout, Flatten, GlobalAveragePooling2D,
#                                                    GlobalMaxPooling2D, MaxPooling2D, Permute,
#                                                    Reshape)
from keras.models import Model,load_model
#from keras.optimizers import RMSprop, SGD
#from keras.preprocessing.image import ImageDataGenerator
#from keras.regularizers import l2
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#from iceburger.io import parse_json_data

PRJ = "/iceburger"
DATA = os.path.join(PRJ, "data/processed")
MODEL = os.path.join(PRJ,"data/model")
TEST ="test.json"
model_path = os.path.join(MODEL,"conv2d_model-0.3184-0.8701.hdf5")
weights_path = os.path.join(MODEL,"conv2d_model-best-0.2949-0.8803-weights.hdf5")
submission_csv_path = "./submission.csv"
batch_size = 32
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


def parse_test_json_data(json_filename):
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
    ID = df.id.values
    X = np.concatenate([image_normalization(x) for x in _X], axis=0)
    X_angle = df.inc_angle.values
    return (ID, X, X_angle)

ID, X_test, X_angle_test = parse_test_json_data(os.path.join(DATA, TEST))

model = load_model(model_path)
model.load_weights(weights_path)
prediction = model.predict(X_test,verbose = 1, batch_size = batch_size)

submission = pd.DataFrame({"id": ID,
                           "is_iceberg": prediction.reshape((
                                        prediction.shape[0]))})
#print(submission.head(10))
submission.to_csv(submission_csv_path, index =False)

