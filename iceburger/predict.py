from __future__ import print_function
from __future__ import division
import numpy as np
import logging
import argparse
import pandas as pd
import sys
import cv2
from keras import backend as K
from keras.models import load_model

FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-predict'

from iceburger.io import image_normalization
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)

"""
PRJ = "/iceburger"
DATA = os.path.join(PRJ, "data/processed")
MODEL = os.path.join(PRJ,"data/model")
TEST = os.path.join(DATA, "test.json")
model_path = os.path.join(MODEL,"conv2d_model-0.3184-0.8701.hdf5")
weights_path = os.path.join(MODEL,"conv2d_model-best-0.2949-0.8803-weights.hdf5")
submission_csv_path = "./submission.csv"
batch_size = 32
"""

def parse_test_json_data(json_filename, percentile=1, padding="avg"):
    """Parse json data to generate trainable matrices
    :param json_filename: path to input json file
    :returns: a `tuple` of
    ID: :class: `numpy.array` of nb_samples of id
    X: :class:`numpy.ndarray` of dimension (nb_samples, height, width, 3)
    X_angle: :class:`numpy.array` of dimension (nb_samples) of incidence
        angles
    """
    df = pd.read_json(json_filename)
    dim = int(np.sqrt(len(df.band_1.iloc[0])))
    _X = np.concatenate([
        np.concatenate([np.array(r.band_1).reshape((dim, dim, 1)),
                        np.array(r.band_2).reshape((dim, dim, 1))],
                       axis=-1)[np.newaxis, :, :, :]
        for _, r in df.iterrows()], axis=0)
    ID = df.id.values
    X = np.concatenate([image_normalization(x, percentile, padding) for x in _X], axis=0)
    X_angle = df.inc_angle.values
    return (ID, X, X_angle)

def predict(args):
    """Making prediction after training

    :param args: arguments as parsed by argparse module
    """
    LOG.info("Loading data from {}".format(args.test))
    ID, X_test, X_angle_test = parse_test_json_data(args.test)
    LOG.info("Loading model from {}".format(args.model_path))
    model = load_model(args.model_path)
    LOG.info("Checking Input Shape...")
    model_input_shape = model.input_shape
    LOG.info("Model_shape is {}".format(model_input_shape))
    if (model_input_shape[1]!=75 or model_input_shape[2]!=75):
        w = model_input_shape[1]
        h = model_input_shape[2]
        X_test = np.array([cv2.resize(x, (w, h)) for x in X_test])
    LOG.info("Start predicting...")
    prediction = model.predict(X_test,verbose = 1, batch_size = args.batch_size)
    submission = pd.DataFrame({"id": ID,
                           "is_iceberg": prediction.reshape((
                                        prediction.shape[0]))})
    #print(submission.head(10))
    LOG.info("Saving prediction to {}".format(args.submission_csv_path))
    submission.to_csv(args.submission_csv_path, index =False)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "model_path", type=str, metavar="MODEL_PATH",
        help="Path to previously saved model")
    parser.add_argument(
        "--batch_size", type=int, metavar="BATCH_SIZE", default=32,
        help="Number of samples in a mini-batch")
    parser.add_argument(
        "--submission_csv_path", type=str, metavar="SUBMISSION_CSV_PATH",
        default="./submission.csv",
        help="Output path where submission of prediction to be saved")
    parser.add_argument(
        "test", type=str, metavar="TEST",
        help=("Path to the json file where test data is saved"))
    args = parser.parse_args()

    prediction = predict(args)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
