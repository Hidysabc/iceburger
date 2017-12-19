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
from iceburger.combined_model_train import create_dataset


FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-predict-copycat'

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

def predict(args):
    """Making prediction after training

    :param args: arguments as parsed by argparse module
    """
    LOG.info("Loading data from {}".format(args.test))
    test = pd.read_json(args.test)
    y_test, X_test_b, X_test_img = create_dataset(args.test, False)
    LOG.info("Loading model from {}".format(args.model_path))
    model = load_model(args.model_path)
    LOG.info("Checking Input Shape...")
    model_input_shape = model.input_shape
    LOG.info("Model_shape is {}".format(model_input_shape))
    """
    if (model_input_shape[1]!=75 or model_input_shape[2]!=75):
        w = model_input_shape[1]
        h = model_input_shape[2]
        X_test = np.array([cv2.resize(x, (w, h)) for x in X_test])
    """
    LOG.info("Start predicting...")
    prediction = model.predict([X_test_b, X_test_img],verbose = 1, batch_size = args.batch_size)
    submission = pd.DataFrame({"id": test["id"],
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
        "--padding", type=str, metavar="PADDING", default="zeros",
        help="Padding arg for parse_test_json_data")
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
