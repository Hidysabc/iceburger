from __future__ import print_function
from __future__ import division

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import shutil
import warnings

from keras import backend as K
from keras import layers
from keras.constraints import non_neg
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.layers import (Input, Activation, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, GlobalAveragePooling2D,
                          AveragePooling2D,
                          GlobalMaxPooling2D, MaxPooling2D, Permute,
                          Reshape, concatenate)
from keras.models import Model, load_model
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from iceburger.io import parse_json_data

FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-composite-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)

#: Number neurons dense layers
NUM_DENSE = 512

#: Dropout ratio
DROPOUT = 0.5

#: Regularization
L2R = 1E-3


def get_callbacks(args,model_out_path):
    """
    Create list of callback functions for fitting step
    :param args: arguments as parsed by argparse module
    :returns: `list` of `keras.callbacks` classes
    """
    checkpoint_name= "{mn}-best_val_loss.hdf5".format(mn=args.model)
    callbacks = []
    #outpath = args.outpath

    # save best model so far
    callbacks.append(
        ModelCheckpoint(
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
    model_configs = json.load(open(args.model_configs, "r"))
    model, optimizer = CompositeModel(model_configs, input_shape=input_shape)
    optimizer = Adam(lr=5e-5)
    #optimizer = SGD(lr=1e-4, momentum=0.9, nesterov=True)
    #optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def CompositeModel(model_configs, input_shape, classes=1):
    inputs = Input(shape=input_shape)
    models = {}
    submodels = []
    for model_name in model_configs:
        models[model_name] = load_model(model_configs[model_name]["path"])
        input_tensor = models[model_name].get_layer(
            model_configs[model_name]["input_layer"]
        ).input
        output_tensor = models[model_name].get_layer(
            model_configs[model_name]["output_layer"]
        ).output
        submodels.append(Model(input_tensor, output_tensor)(inputs))

    x = Concatenate()(submodels)
    optimizer = models["resnet"].optimizer
    x = Dense(NUM_DENSE, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(NUM_DENSE, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    output = Dense(classes, activation="sigmoid",
                   kernel_regularizer=l2(L2R),
                   bias_regularizer=l2(L2R),
                   name="predictions")(x)

    # Create imodel.
    model = Model(input=inputs, output=output, name = "composite")

    model.summary()
    return model, optimizer


def train(args):
    """Train a neural network

    :param args: arguments as parsed by argparse module
    """
    LOG.info("Loading data from {}".format(args.data))
    X, X_angle, y, subset = parse_json_data(os.path.join(args.data))
    X_train = X[subset=='train']
    X_angle_train = X_angle[subset=='train']
    y_train = y[subset=='train']
    X_valid = X[subset=='valid']
    X_angle_valid = X_angle[subset=='valid']
    y_valid = y[subset=='valid']

    LOG.info("Initiate model")
    model = compile_model(args, input_shape=(75,75,3))

    LOG.info("Create sample generators")
    gen_train = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         channel_shift_range = 0,
                         zoom_range = 0.2,
                         rotation_range = 30)

    gen_valid = ImageDataGenerator(horizontal_flip = False,
                         vertical_flip = False,
                         width_shift_range = 0.0,
                         height_shift_range = 0.0,
                         channel_shift_range = 0,
                         zoom_range = 0.0,
                         rotation_range = 0)

    # Here is the function that merges our two generators
    # We use the exact same generator with the same random seed for both the y and angle arrays
    """
    def gen_flow_for_one_input(X1, y1):
        genX1 = gen_train.flow(X1, y1, batch_size= args.batch_size, seed=666)
        while True:
            X1i = genX1.next()
            yield X1i[0], X1i[1]
    """
    #g_seed = int(input("Please provide a random integer ranging from 1 to 100000:"))
    g_seed = np.random.randint(0,10000)
    def gen_flow_train_for_two_input(X1, X2, y1):
        genX1 = gen_train.flow(X1, y1, batch_size= args.batch_size, seed=g_seed)
        genX2 = gen_train.flow(X1, X2, batch_size = args.batch_size, seed=g_seed)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]


    #Finally create out generator
    gen_train_ = gen_train.flow(X_train, y_train, batch_size = args.batch_size, seed=g_seed, shuffle=True)
    gen_valid_ = gen_valid.flow(X_valid, y_valid, batch_size = args.batch_size, seed=g_seed, shuffle=False)
    #gen_train_ = gen_flow_train_for_two_input(X_train, X_angle_train, y_train)
    #gen_valid_ = gen_flow_train_for_two_input(X_valid, X_angle_valid, y_valid)


    LOG.info("Create callback functions")
    model_out_path = os.path.join(os.path.abspath(args.outpath),"checkpoints")
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

    best_model_name = "{mn}-best-{val_loss:.4f}-{val_acc:.4f}-model.hdf5".format(
        mn=args.model,
        val_loss=history.history["val_loss"][best_idx],
        val_acc=history.history["val_acc"][best_idx]
    )
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
        "--model", type=str, metavar="MODEL", default= "composite",
        help="Model type for training (Options: composite)")
    parser.add_argument(
        "--model_configs", type=str, metavar="MODEL_CONFIG", default=None,
        help="Path to a config file in .json indicating which model to use")
    parser.add_argument(
        "--batch_size", type=int, metavar="BATCH_SIZE", default=64,
        help="Number of samples in a mini-batch")
    parser.add_argument(
        "--epochs", type=int, metavar="EPOCHS", default=1000,
        help="Number of epochs")
    parser.add_argument(
        "--train_steps", type=int, metavar="TRAIN_STEPS", default=64,
        help=("Number of mini-batches for each epoch to pass through during"
              " training"))
    parser.add_argument(
        "--valid_steps", type=int, metavar="VALID_STEPS", default=64,
        help=("Number of mini-batches for each epoch to pass through during"
              " validation"))
    parser.add_argument(
        "--cb_early_stop", type=int, metavar="PATIENCE", default=50,
        help="Number of epochs for early stop if without improvement")
    parser.add_argument(
        "--cb_reduce_lr", type=int, metavar="PLATEAU", default=50,
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
