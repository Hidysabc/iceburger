from __future__ import print_function
from __future__ import division
import numpy as np
import os
import logging
import argparse
import pandas as pd
import sys
import shutil

from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.layers import (Input, Activation, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, MaxPooling2D, Permute,
                          Reshape)
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from iceburger.io import parse_json_data

FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-resnet50-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)

PRJ = "/workspace/iceburger"
DATA = os.path.join(PRJ, "data/processed")

#: Number filters convolutional layers
NUM_CONV_FILTERS = 64

#: Size filters convolutional layers
CONV_FILTER_ROW = 3
CONV_FILTER_COL = 3

#: Dropout ratio
DROPOUT = 0.5

#: Regularization
L1L2R = 1E-3
L2R = 5E-3

def get_callbacks(args,model_out_path):
    """
    Create list of callback functions for fitting step
    :param args: arguments as parsed by argparse module
    :returns: `list` of `keras.callbacks` classes
    """
    checkpoint_name= "{mn}-best_val_loss_weights.hdf5".format(mn=args.model)
    callbacks = []
    #outpath = args.outpath

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
    if args.model.lower()=="conv2d_model":
        model = Conv2D_Model(weights=args.weights, include_top=True,
                              input_shape=input_shape)
        optimizer = SGD(lr = 0.0001, momentum = 0.9)
    else:
        LOG.err("Unknown model name: {}".format(args.model))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



def conv2d_bn(x, filters, num_row, num_col, padding = 'same',
              strides = (1,1), name = 'None'):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def Conv2D_Model(include_top = True, weights=None, input_tensor = None,
         input_shape = None, pooling = None, classes = 1):
    """Instantiate the Conv architecture

    :param include_top: whether to include fully-connected layers at the top
        of the network
    :param weights: path to pretrained weights or `None`
    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.
    :param input_shape: optional shape tuple, only to be specified if
        `include_top` is False
    :param pooling: optional pooling mode for feature extraction when
        `include_top` is False
    :param classes: optional number of classes to classify data into, only to
        be specified if `include_top` is True, and if no `weights` argument
        is specified.
    :returns: A :class:`keras.models.Model` instance
    """
    if input_tensor is None:
        image_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            image_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            image_input = input_tensor

    x = conv2d_bn(image_input, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn1")
    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn2")
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn3")
    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn4")
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn5")
    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn6")
    x = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn7")
    x = conv2d_bn(x, NUM_CONV_FILTERS, CONV_FILTER_ROW, CONV_FILTER_COL, strides=(1,1),
                  name = "conv2d_bn8")
    x = MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    #x = Dropout(DROPOUT, name="drop1")(x)

    if include_top:
        x = Dense(classes, activation="sigmoid",
                  W_regularizer=l2(L2R),
                  b_regularizer=l2(L2R),
                  name="predictions")(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = image_input
    # Create model.
    model = Model(inputs, x, name='conv2d_model')
    if weights:
        model.load_weights(weights)

    return model

def train(args):
    """Train a neural network

    :param args: arguments as parsed by argparse module
    """
    LOG.info("Loading data from {}".format(DATA))
    X, X_angle, y, subset = parse_json_data(os.path.join(DATA, "train_valid.json"))
    #w = 75
    #h = 75
    X_train = X[subset=='train']
    X_angle_train = X_angle[subset=='train']
    y_train = y[subset=='train']
    X_valid = X[subset=='valid']
    X_angle_valid = X_angle[subset=='valid']
    y_valid = y[subset=='valid']
    #ds = DataSet.from_pickle(args.data)
    #nb_classes = ds.df.activity.nunique()

    LOG.info("Initiate model")
    model = compile_model(args, input_shape=(75,75,3))

    LOG.info("Create sample generators")
    gen_train = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 45)

    gen_valid = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 45)
    # Here is the function that merges our two generators
    # We use the exact same generator with the same random seed for both the y and angle arrays
    def gen_flow_for_one_input(X1, y):
        genX1 = gen_train.flow(X1, y, batch_size= args.batch_size, seed=666)
        while True:
            X1i = genX1.next()
            yield X1i[0], X1i[1]

    #Finally create out generator
    gen_train_ = gen_train.flow(X_train, y_train, batch_size = args.batch_size, seed=666)
    gen_valid_ = gen_valid.flow(X_valid, y_valid, batch_size = args.batch_size, seed=666)
    #gen_train_ = gen_flow_train_for_one_input(X_train, y_train)
    #gen_valid_ = gen_flow_valid_for_one_input(X_valid, y_valid)


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

    best_model_name = "{mn}-best-{val_loss:.4f}-{val_acc:.4f}-weights.hdf5".format(
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
    model.save(os.path.join(args.outpath, final_file_root))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model", type=str, metavar="MODEL", default= "conv2d_model",
        help="Model type for training (Options: resnet50)")
    parser.add_argument(
        "--weights", type=str, metavar="WEIGHTS", default=None,
        help="Path to previously saved weights")
    parser.add_argument(
        "--batch_size", type=int, metavar="BATCH_SIZE", default=32,
        help="Number of samples in a mini-batch")
    parser.add_argument(
        "--epochs", type=int, metavar="EPOCHS", default=1000,
        help="Number of epochs")
    parser.add_argument(
        "--train_steps", type=int, metavar="TRAIN_STEPS", default=512,
        help=("Number of mini-batches for each epoch to pass through during"
              " training"))
    parser.add_argument(
        "--valid_steps", type=int, metavar="VALID_STEPS", default=128,
        help=("Number of mini-batches for each epoch to pass through during"
              " validation"))
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

    model = train(args)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
