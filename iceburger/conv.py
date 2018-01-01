from __future__ import print_function
from __future__ import division
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.layers import (Input, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Model


#: Number filters convolutional layers
NUM_CONV_FILTERS = 128

#: Size filters convolutional layers
CONV_FILTER_ROW = 3
CONV_FILTER_COL = 3

#: Number neurons dense layers
NUM_DENSE = 512

#: Dropout ratio
DROPOUT = 0.5
DROPOUT_fcnn = 0.2

#: Regularization
L1L2R = 1E-3
L2R = 1E-3


def Conv2DModel(include_top=True, weights=None, input_tensor=None,
                input_shape=None, pooling=None, classes=1):
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

    fcnn = Conv2D(32, kernel_size=(3, 3), activation="relu")(
        BatchNormalization()(image_input))
    fcnn = MaxPooling2D((3, 3))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = Conv2D(64, kernel_size=(3, 3), activation="relu")(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = Conv2D(128, kernel_size=(3, 3), activation="relu")(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = Conv2D(128, kernel_size=(3, 3), activation="relu")(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = BatchNormalization()(fcnn)
    fcnn = Flatten(name="flatten")(fcnn)
    # local_input = image_input
    # partial_model = Model(image_input, fcnn)
    dense = Dropout(DROPOUT_fcnn)(fcnn)
    dense = Dense(256, activation="relu")(dense)
    dense = Dropout(DROPOUT_fcnn)(dense)
    dense = Dense(128, activation="relu")(dense)
    dense = Dropout(DROPOUT_fcnn)(dense)
    dense = Dense(64, activation="relu")(dense)
    dense = Dropout(DROPOUT_fcnn)(dense)
    # For some reason i've decided not to normalize angle data
    output = Dense(1, activation="sigmoid")(dense)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = image_input

    model = Model(inputs, output)
    # Create model.
    if weights:
        model.load_weights(weights)

    return model
