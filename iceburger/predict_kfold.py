from __future__ import print_function
from __future__ import division
import numpy as np
import logging
import argparse
import os
import pandas as pd
import sys
import cv2
from keras.models import load_model, model_from_json
from .io import parse_json_data
from .ensemble import KFoldEnsembleKerasModel
from .filters import gabor_filter, sobel_filter
FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-predict'

# from iceburger.io import image_normalization
logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)


def predict(args):
    """Making prediction after training

    :param args: arguments as parsed by argparse module
    """
    LOG.info("Loading model from {}".format(args.model_arch_path))
    model_arch_path = args.model_arch_path
    model_dir = os.path.dirname(model_arch_path)
    model_name = os.path.basename(model_arch_path).split("-", 1)[0]
    model = KFoldEnsembleKerasModel(model_arch_path)

    LOG.info("Load model weights from {}".format(model_dir))
    model.load_weights(os.path.join(model_dir, model_name))

    LOG.info("Loading data from {}".format(args.test))
    ID, X_test, X_angle_test, y, _ = parse_json_data(args.test,
                                                padding=args.padding,
                                                smooth=args.smooth,
                                                filter_instance=gabor_filter,
                                                frequency=0.6, theta=np.pi*90/180,
                                                mode="nearest")
    """
    ID, X_test, X_angle_test, y, _ = parse_json_data(
        args.test, padding=args.padding, smooth=args.smooth)
    """
    LOG.info("Start predicting...")
    prediction = model.predict(X_test, verbose=1, batch_size=args.batch_size)
    submission = pd.DataFrame({"id": ID,
                               "is_iceberg": prediction.reshape((
                                             prediction.shape[0]))})
    # print(submission.head(10))
    LOG.info("Saving prediction to {}".format(args.submission_csv_path))
    submission.to_csv(args.submission_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "model_arch_path", type=str, metavar="MODEL_ARCH_PATH",
        help="Path to model architecture json file")
    parser.add_argument(
        "--padding", type=str, metavar="PADDING", default="zeros",
        help="Padding arg for parse_test_json_data")
    parser.add_argument(
        "--smooth", type=bool, metavar="SMOOTH", default=True,
        help="Whether to perform band smooth on data")
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

    predict(args)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
