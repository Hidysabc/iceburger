"""
Function for training model
"""

import argparse
import logging
import numpy as np
import os
import sys
import shutil

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from .conv import Conv2DModel
from .io import parse_json_data


FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-conv2d-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)


def get_callbacks(args, model_out_path):
    """
    Create list of callback functions for fitting step
    :param args: arguments as parsed by argparse module
    :returns: `list` of `keras.callbacks` classes
    """
    checkpoint_name = "{mn}-best_val_loss.hdf5".format(mn=args.model)
    callbacks = []
    # outpath = args.outpath

    # save best model so far
    callbacks.append(
        ModelCheckpoint(
            filepath=os.path.join(model_out_path, checkpoint_name),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )
    )
    if args.cb_early_stop:
        # stop training earlier if the model is not improving
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.cb_early_stop,
                verbose=1, mode='auto'
            )
        )
    if args.cb_reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.cb_reduce_lr_factor,
                patience=args.cb_reduce_lr,
                min_lr=1e-8
            )
        )

    return callbacks, checkpoint_name


def compile_model(args, input_shape):
    """Build and compile model

    :param args: arguments as parsed by argparse module
    :returns: `keras.models.Model` of compiled model
    """
    lr = args.lr
    if args.model.lower() == "conv2d_model":
        if args.model_path:
            model = load_model(args.model_path)
        else:
            model = Conv2DModel(include_top=True,
                                input_shape=input_shape)
        optimizer = Adam(lr, decay=1e-6)
        # optimizer = SGD(lr = 0.001, momentum = 0.9)
    else:
        LOG.err("Unknown model name: {}".format(args.model))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train(args):
    """Train a neural network

    :param args: arguments as parsed by argparse module
    """
    LOG.info("Loading data from {}".format(args.data))
    ID, X, X_angle, y, subset = parse_json_data(os.path.join(args.data),
                                                padding=args.padding,
                                                smooth=args.smooth)
    X_train = X[subset == 'train']
    X_angle_train = X_angle[subset == 'train']
    y_train = y[subset == 'train']
    X_valid = X[subset == 'valid']
    X_angle_valid = X_angle[subset == 'valid']
    y_valid = y[subset == 'valid']

    LOG.info("Initiate model")
    model = compile_model(args, input_shape=(75, 75, 3))
    LOG.info("Save model structure")
    model_arch_name = "{mn}-arch.json".format(mn=args.model)
    with open(os.path.join(args.outpath, model_arch_name), "w") as fo:
        fo.write(model.to_json())

    LOG.info("Create sample generators")
    gen_train = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.,
                                   height_shift_range=0.,
                                   channel_shift_range=0.,
                                   zoom_range=0.2,
                                   rotation_range=10)

    # gen_valid = ImageDataGenerator(horizontal_flip=True,
    #                                vertical_flip=True,
    #                                width_shift_range=0.1,
    #                                height_shift_range=0.1,
    #                                zoom_range=0.1,
    #                                rotation_range=45)
    # Here is the function that merges our two generators
    # We use the exact same generator with the same random seed for both the y
    # and angle arrays

    def gen_flow_for_one_input(X1, y):
        genX1 = gen_train.flow(X1, y, batch_size=args.batch_size, seed=666)
        while True:
            X1i = genX1.next()
            yield X1i[0], X1i[1]

    # Finally create out generator
    gen_train_ = gen_train.flow(X_train, y_train, batch_size=args.batch_size,
                                seed=666)
    # gen_valid_ = gen_valid.flow(X_valid, y_valid, batch_size=args.batch_size,
    #                             seed=666)
    # gen_train_ = gen_flow_train_for_one_input(X_train, y_train)
    # gen_valid_ = gen_flow_valid_for_one_input(X_valid, y_valid)

    LOG.info("Create callback functions")
    model_out_path = os.path.join(os.path.abspath(args.outpath), "checkpoints")
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    callbacks, checkpoint_model_name = get_callbacks(args, model_out_path)

    LOG.info("Start training ...")
    history = model.fit_generator(
        gen_train_,
        steps_per_epoch=X_train.shape[0] / args.batch_size,
        epochs=args.epochs, verbose=1,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks)

    best_idx = np.argmin(history.history['val_loss'])
    LOG.info("Best model occurred at epoch {}:".format(best_idx))
    LOG.info("  Best validation score: {:.4f}".format(
        history.history["val_loss"][best_idx])
    )
    LOG.info("  Best validation accuracy: {:.4f} ".format(
        history.history["val_acc"][best_idx])
    )

    best_model_name = "{mn}-best-{val_loss:.4f}-{val_acc:.4f}-weights.hdf5"\
        .format(mn=args.model, val_loss=history.history["val_loss"][best_idx],
                val_acc=history.history["val_acc"][best_idx])
    LOG.info("Rename best model to {}".format(best_model_name))
    shutil.move(
        os.path.join(model_out_path, checkpoint_model_name),
        os.path.join(args.outpath, best_model_name)
    )

    LOG.info("Saving final model ...")
    LOG.info("  Final validation score: {:.4f}".format(
        history.history["val_loss"][-1])
    )
    LOG.info("  Final validation accuracy: {:.4f} ".format(
        history.history["val_acc"][-1])
    )

    final_file_root = "{mn}-{val_loss:.4f}-{val_acc:.4f}".format(
        mn=args.model,
        val_loss=history.history["val_loss"][-1],
        val_acc=history.history["val_acc"][-1]
    )
    model.save(os.path.join(args.outpath, final_file_root+'.hdf5'))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data", type=str, metavar="DATA",
        help=("Path to training data stored in json."))
    parser.add_argument(
        "--padding", type=str, metavar="PADDING", default="zeros",
        help="Padding arg for parse_test_json_data")
    parser.add_argument(
        "--smooth", type=bool, metavar="SMOOTH", default=True,
        help="Whether to perform band smooth on data")
    parser.add_argument(
        "--model", type=str, metavar="MODEL", default="conv2d_model",
        help="Model type for training (Options: conv2d_model)")
    parser.add_argument(
        "--model_path", type=str, metavar="MODEL_PATH", default=None,
        help="Path to previously saved model (*.hdf5)")
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
        "--epochs", type=int, metavar="EPOCHS", default=100,
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

    train(args)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
