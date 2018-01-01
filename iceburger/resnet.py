from __future__ import print_function
from __future__ import division
from keras import backend as K
# from keras import layers
# from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
from keras.utils import layer_utils
from keras.engine.topology import get_source_inputs
from keras.layers import (Input, Activation, BatchNormalization,
                          Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D,
                          AveragePooling2D,
                          GlobalMaxPooling2D, MaxPooling2D)
from keras.models import Model, load_model
from keras.regularizers import l2


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
L2R = 5E-3


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer
        at shortcut.
    # Arguments
    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle
        conv layer at main path
    :param filters: list of integers, the filters of 3 conv
        layer at main path
    :param stage: integer, current stage label, used for
        generating layer names
    :param block: 'a','b'..., current block label, used for
        generating layer names
    # Returns
    :returns: Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle
        conv layer at main path
    :param filters: list of integers, the filters of 3 conv layer
        at main path
    :param stage: integer, current stage label, used for
        generating layer names
    :param block: 'a','b'..., current block label, used for
        generating layer names
    # Returns
    :returns: Output tensor for the block.
    # Note that from stage 3, the first conv layer at main path
        is with strides=(2,2). And the shortcut should have
        strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), strides=strides,
              name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNetModel(include_top=True, weights=None, input_tensor=None,
           input_shape=None, pooling=None, classes=1, stage=2):
    """Instantiates the ResNet architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
    :param include_top: whether to include the fully-connected
        layer at the top of the network.
    :param weights: one of `None` (random initialization)
        or 'imagenet' (pre-training on ImageNet).
    :param input_tensor: optional Keras tensor (i.e. output of
        `layers.Input()`) to use as image input for the model.
    :param input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last`
        data format) or `(3, 224, 224)` (with `channels_first`
        data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 197.
        E.g. `(200, 200, 3)` would be one valid value.
    :param pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the last convolutional layer.
        - `avg` means that global average pooling will be applied
            to the output of the last convolutional layer, and
            thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    :param classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
    :param stage: which stage shall the ResNet model be formed
    # Returns
    :returns: A Keras model instance.
    # Raises
    :raises: ValueError: in case of invalid argument for
        `weights`, or invalid input shape.
    """

    if input_tensor is None:
        image_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            image_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            image_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    resnet = Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                    name='conv1')(image_input)
    resnet = BatchNormalization(axis=bn_axis,
                                name='bn_conv1')(resnet)
    resnet = Activation('relu')(resnet)
    resnet = MaxPooling2D((3, 3), strides=(2, 2))(resnet)
    if stage > 1:
        resnet = conv_block(resnet, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        resnet = identity_block(resnet, 3, [64, 64, 256], stage=2, block='b')
        resnet = identity_block(resnet, 3, [64, 64, 256], stage=2, block='c')
    if stage > 2:
        resnet = conv_block(resnet, 3, [128, 128, 512], stage=3, block='a')
        resnet = identity_block(resnet, 3, [128, 128, 512], stage=3, block='b')
        resnet = identity_block(resnet, 3, [128, 128, 512], stage=3, block='c')
        resnet = identity_block(resnet, 3, [128, 128, 512], stage=3, block='d')
    if stage > 3:
        resnet = conv_block(resnet, 3, [256, 256, 1024], stage=4, block='a')
        resnet = identity_block(resnet, 3, [256, 256, 1024], stage=4, block='b')
        resnet = identity_block(resnet, 3, [256, 256, 1024], stage=4, block='c')
        resnet = identity_block(resnet, 3, [256, 256, 1024], stage=4, block='d')
        resnet = identity_block(resnet, 3, [256, 256, 1024], stage=4, block='e')
        resnet = identity_block(resnet, 3, [256, 256, 1024], stage=4, block='f')
    if stage > 4:
        resnet = conv_block(resnet, 3, [512, 512, 2048], stage=5, block='a')
        resnet = identity_block(resnet, 3, [512, 512, 2048], stage=5, block='b')
        resnet = identity_block(resnet, 3, [512, 512, 2048], stage=5, block='c')
    if include_top:
        resnet = GlobalAveragePooling2D()(resnet)
        # x = GlobalMaxPooling2D()(x)
        dense = Dense(512, name="fc_1")(resnet)
        dense = BatchNormalization(name = "bn_fc1")(dense)
        dense = Activation("relu")(dense)
        dense = Dropout(DROPOUT)(dense)
        dense = Dense(256, activation="relu")(dense)
        dense = Dropout(DROPOUT)(dense)
        dense = Dense(128, activation="relu")(dense)
        dense = Dropout(DROPOUT)(dense)
        output = Dense(classes, activation="sigmoid",
                       name="predictions")(dense)
    else:
        if pooling == 'avg':
            output = GlobalAveragePooling2D()(resnet)
        elif pooling == 'max':
            output = GlobalMaxPooling2D()(resnet)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = image_input
    # Create imodel.
    model = Model(input=inputs, output=output, name="resnet")
    # print(model.summary())
    if weights:
        model.load_weights(weights)
    return model


