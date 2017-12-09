from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import os
import logging
import argparse
import sys
import shutil

from keras import backend as K
from keras.layers import (Input, Activation, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, AveragePooling2D,
                          GlobalAveragePooling2D,
                          GlobalMaxPooling2D, MaxPooling2D, Permute,
                          Reshape)
from keras.models import Model,load_model
from keras.optimizers import Adam, Nadam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.applications import InceptionV3

import warnings
import boto3

from iceburger.io import parse_json_data
from iceburger.callbacks import ModelCheckpointS3


FORMAT = '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-inception-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)

s3bucket = "iceburger"
input_dir = '/tmp/iceburger/'
s3 = boto3.resource('s3')
bucket = s3.Bucket(s3bucket)
s3_client = boto3.client('s3')

#: Number filters convolutional layers
NUM_CONV_FILTERS = 128

#: Size filters convolutional layers
CONV_FILTER_ROW = 3
CONV_FILTER_COL = 3

#: Number neurons dense layers
NUM_DENSE = 512

#: Dropout ratio
DROPOUT = 0.5

#: Regularization
L1L2R = 1E-3
L2R = 1E-3


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
        ModelCheckpointS3(
            filepath=os.path.join(model_out_path, checkpoint_name),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False
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
    # lr = 1e-3
    # LOG.info("Learning rate: {}".format(lr))
    if args.model.lower() == "inception_model":
        if args.model_path:
            model_path = os.path.join(input_dir, args.model_path)
            s3_client.download_file(s3bucket, args.model_path, model_path)
            model = load_model(args.model_path)
        else:
            LOG.info("Use imagenet weights")
            inception = InceptionV3(include_top=False, input_shape=input_shape)
            last_mix = inception.get_layer(name="mixed10")
            top_layer = GlobalAveragePooling2D(name='avg_pool')(last_mix.output)
            top_layer = Dense(1, activation="sigmoid",
                              kernel_regularizer=l2(L2R),
                              bias_regularizer=l2(L2R),
                              name="predictions")(top_layer)
            model = Model(inputs=inception.input, outputs=top_layer)
            print(model.summary())
        optimizer = Nadam()
        #optimizer = SGD(lr=1e-5, decay=1e-4, momentum=0.9, nesterov=True)
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
    data_path = os.path.join(input_dir, args.data)
    s3_client.download_file(s3bucket, args.data, data_path)
    LOG.info("Loading data from {}".format(data_path))
    X, X_angle, y, subset = parse_json_data(os.path.join(data_path),padding="avg")


    #LOG.info("Loading data from {}".format(args.data))
    #X, X_angle, y, subset = parse_json_data(os.path.join(args.data),
    #                                        padding="avg")
    w = 299
    h = 299
    X_train = np.array([cv2.resize(x, (w, h)) for x in X[subset == 'train']])
    X_angle_train = X_angle[subset == 'train']
    y_train = y[subset == 'train']
    #Y_train = to_categorical(y_train, 2)
    X_valid = np.array([cv2.resize(x, (w, h)) for x in X[subset == 'valid']])
    X_angle_valid = X_angle[subset == 'valid']
    y_valid = y[subset == 'valid']
    #Y_valid = to_categorical(y_valid, 2)

    LOG.info("Initiate model")
    model = compile_model(args, input_shape=(299, 299, 3))

    LOG.info("Create sample generators")
    gen_train = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=30)

    gen_valid = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=30)

    g_seed = np.random.randint(1,10000)

    def gen_flow_for_one_input(X1, y):
        genX1 = gen_train.flow(X1, y, batch_size=args.batch_size, seed=g_seed)
        while True:
            X1i = genX1.next()
            yield X1i[0], X1i[1]

    # Finally create out generator
    gen_train_ = gen_train.flow(X_train, y_train,
                                batch_size=args.batch_size, seed=g_seed)
    gen_valid_ = gen_valid.flow(X_valid, y_valid,
                                batch_size=args.batch_size, seed=g_seed)
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
        steps_per_epoch=args.train_steps,
        epochs=args.epochs, verbose=1,
        validation_data=gen_valid_,
        validation_steps=args.valid_steps,
        callbacks=callbacks)

    best_idx = np.argmin(history.history['val_loss'])
    LOG.info("Best model occurred at epoch {}:".format(best_idx))
    LOG.info("  Best validation score: {:.4f}".format(
        history.history["val_loss"][best_idx])
    )
    LOG.info("  Best validation accuracy: {:.4f} ".format(
        history.history["val_acc"][best_idx])
    )

    best_model_name = "{mn}-best-{val_loss:.4f}-{val_acc:.4f}-model.hdf5"\
        .format(mn=args.model,
                val_loss=history.history["val_loss"][best_idx],
                val_acc=history.history["val_acc"][best_idx])
    LOG.info("Rename best model to {}".format(best_model_name))
    shutil.move(
        os.path.join(model_out_path, checkpoint_model_name),
        os.path.join(args.outpath, best_model_name)
    )
    copy_source = {
                    "Bucket": s3bucket,
                    "Key": "{mn}-best_val_loss.hdf5".format(mn=args.model)

    }

    s3_client.copy(copy_source, s3bucket, best_model_name)

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
    s3_client.upload_file(os.path.join(args.outpath, final_file_root+'.hdf5'), s3bucket, final_file_root+'.hdf5')



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data", type=str, metavar="DATA",
        help=("Path to training data stored in json."))
    parser.add_argument(
        "--model", type=str, metavar="MODEL", default="inception_model",
        help="Model type for training (Options: inception_model)")
    parser.add_argument(
        "--model_path", type=str, metavar="MODEL_PATH", default=None,
        help="Path to previously saved model(*.hdf5)")
    parser.add_argument(
        "--batch_size", type=int, metavar="BATCH_SIZE", default=8,
        help="Number of samples in a mini-batch")
    parser.add_argument(
        "--epochs", type=int, metavar="EPOCHS", default=1000,
        help="Number of epochs")
    parser.add_argument(
        "--train_steps", type=int, metavar="TRAIN_STEPS", default=2048,
        help=("Number of mini-batches for each epoch to pass through during"
              " training"))
    parser.add_argument(
        "--valid_steps", type=int, metavar="VALID_STEPS", default=512,
        help=("Number of mini-batches for each epoch to pass through during"
              " validation"))
    parser.add_argument(
        "--cb_early_stop", type=int, metavar="PATIENCE", default=50,
        help="Number of epochs for early stop if without improvement")
    parser.add_argument(
        "--cb_reduce_lr", type=int, metavar="PLATEAU", default=3,
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
