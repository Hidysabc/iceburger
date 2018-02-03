"""
Function for training k-fold ensemble model
"""

import argparse
import logging
import numpy as np
import os
import sys

from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from iceburger.conv import Conv2DModel
from iceburger.resnet import ResNet
from iceburger.inception import Inception
from iceburger.ensemble_angle import KFoldEnsembleKerasModel
from iceburger.io import parse_json_data
from iceburger.train import get_callbacks
from iceburger.filters import gabor_filter, sobel_filter

FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-kfold-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)


def get_model_arch_path(args, input_shape):
    if args.model.lower() == "conv2d_model":
        model = Conv2DModel(include_top=True,
                            input_shape=input_shape)
    elif args.model.lower() == "resnet":
        model = ResNet(include_top=True,
                       input_shape=input_shape,
                       stage=2)
    elif args.model.lower() == "inception":
        model = Inception(include_top=True,
                          input_shape=input_shape,
                          mixed=2)
    else:
        LOG.err("Unknown model name: {}".format(args.model))

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    model_arch_path = os.path.join(args.outpath,
                                   "{}-arch.json".format(args.model))
    with open(os.path.join(args.outpath, model_arch_path), "w") as fo:
        fo.write(model.to_json())

    return model_arch_path


def train_kfold(args):
    LOG.info("Loading data from {}".format(args.data))
    ID, X, X_angle, y, subset = parse_json_data(os.path.join(args.data),
                                                padding=args.padding,
                                                smooth=args.smooth,
                                                filter_instance=gabor_filter,
                                                frequency=0.6, theta=np.pi*90/180,
                                                mode="nearest")
    LOG.info("Create sample generators")
    gen_train = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.,
                                   height_shift_range=0.,
                                   channel_shift_range=0.,
                                   zoom_range=0.2,
                                   rotation_range=15)
    """
    def gen_flow_for_two_input(X1, X2, y):
        genX1 = gen_train.flow(X1, y, batch_size=args.batch_size, seed=666)
        genX2 = gen_train.flow(X1, X2, batch_size=args.batch_size, seed=666)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]

    gen_train_ = gen_flow_for_two_input(X, X_angle, y)
    """
    LOG.info("Create callback functions")
    model_out_path = args.outpath
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    callbacks, _ = get_callbacks(args, model_out_path)
    checkpoint_name = os.path.join(model_out_path, args.model)

    n_splits = args.n_splits
    n_repeats = args.n_repeats
    LOG.info("K-fold training: splits:{}, repeats:{}"
             .format(n_splits, n_repeats))
    model_arch_path = get_model_arch_path(args, input_shape=(75, 75, 14))

    model = KFoldEnsembleKerasModel(model_arch_path, n_splits=n_splits,
                                    n_repeats=n_repeats, random_states=913)

    lr = args.lr
    decay = args.lrdecay
    kfold_info = model.train_generator(
        X, X_angle, y, gen_train, batch_size=args.batch_size,
        checkpoint_name=checkpoint_name,
        epochs=args.epochs,
        callbacks=callbacks, loss="binary_crossentropy",
        optimizer_class=Adam, lr=lr, decay=decay,
        metrics=["accuracy"])

    val_losses = [kfold_info[split]["best_val_loss"]
                  for split in kfold_info]
    best_epochs = [kfold_info[split]["best_epoch"]
                   for split in kfold_info]
    avg_val_loss = np.mean(val_losses)

    LOG.info("Val losses: {}".format(val_losses))
    LOG.info("Best epochs: {}".format(best_epochs))
    LOG.info("Average k-fold val loss: {}".format(avg_val_loss))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data", type=str, metavar="DATA",
        help=("Path to training data stored in json."))
    parser.add_argument(
        "--model", type=str, metavar="MODEL", default="conv2d_model",
        help="Model type for training (Options: conv2d_model)")
    parser.add_argument(
        "--padding", type=str, metavar="PADDING", default="zeros",
        help="Padding arg for parse_test_json_data")
    parser.add_argument(
        "--smooth", type=bool, metavar="SMOOTH", default=True,
        help="Whether to perform band smooth on data")
    parser.add_argument(
        "--n_splits", type=int, metavar="NUM_SPLITS", default=5,
        help="Number of splits")
    parser.add_argument(
        "--n_repeats", type=int, metavar="NUM_REPEATS", default=3,
        help="Number of repeated splits")
    parser.add_argument(
        "--batch_size", type=int, metavar="BATCH_SIZE", default=32,
        help="Number of samples in a mini-batch")
    parser.add_argument(
        "--lr", type=int, metavar="LR", default=1e-3,
        help="learning rate")
    parser.add_argument(
        "--lrdecay", type=int, metavar="LR_DECAY", default=1e-6,
        help="learning rate decay")
    parser.add_argument(
        "--epochs", type=int, metavar="EPOCHS", default=500,
        help="Number of epochs")
    parser.add_argument(
        "--cb_early_stop", type=int, metavar="PATIENCE", default=50,
        help="Number of epochs for early stop if without improvement")
    parser.add_argument(
        "--cb_reduce_lr", type=int, metavar="PLATEAU", default=10,
        help="Number of epochs to reduce learning rate without improvement")
    parser.add_argument(
        "--cb_reduce_lr_factor", type=float, metavar="ALPHA", default=0.5,
        help=("Factor for reducing learning rate. Only activated when"
              " `cb_reduce_lr` is set"))
    parser.add_argument(
        "--outpath", type=str, metavar="OUTPATH",
        default="./",
        help="Output path where parsed data set class to be saved")
    args = parser.parse_args()

    train_kfold(args)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
