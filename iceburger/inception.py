from __future__ import print_function
from __future__ import division
from keras import backend as K
from keras import layers
from keras.engine.topology import get_source_inputs
from keras.layers import (Input, Activation, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, AveragePooling2D,
                          GlobalAveragePooling2D,
                          GlobalMaxPooling2D, MaxPooling2D)
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape

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

def conv2d_bn(x, filters, num_row, num_col, padding='same',
              strides=(1,1), name=None):
    """Utility function to apply conv + BN.
    # Arguments
    :param x: input tensor.
    :param filters: filters in `Conv2D`.
    :param num_row: height of the convolution kernel.
    :param num_col: width of the convolution kernel.
    :param padding: padding mode in `Conv2D`.
    :param strides: strides in `Conv2D`.
    :param name: name of the ops; will become `name + '_conv'`
        for the convolution and `name + '_bn'` for the batch
        norm layer.
    # Returns
    :returns Output tensor after applying `Conv2D` and `BatchNormalization`.
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

def InceptionModel(include_top=True, weights=None,
                   input_tensor=None, input_shape=None,
                   pooling=None, classes=1, mixed=2):
    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is
    299x299.
    # Arguments
    :param include_top: whether to include the fully-connected
    :param layer at the top of the network.
    :param weights: one of `None` (random initialization)
        or 'imagenet' (pre-training on ImageNet).
    :param input_tensor: optional Keras tensor (i.e. output of
        `layers.Input()`) to use as image input for the model.
    :param input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape has
        to be `(299, 299, 3)` (with `channels_last` data format)
        or `(3, 299, 299)` (with `channels_first` data format).
        It should have exactly 3 inputs channels, and width and
        height should be no smaller than 139.
        E.g. `(150, 150, 3)` would be one valid value.
    :param pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be the
            4D tensor output of the last convolutional layer.
        - `avg` means that global average pooling will be applied
            to the output of the last convolutional layer, and
            thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    :param classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
    :param mixed: which mixed stage for inception model
    # Returns
    :returns A Keras model instance.
    # Raises
    :raises ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
    """
    if input_tensor is None:
        image_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            image_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            image_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # inception = conv2d_bn(image_input, 32, 3, 3, strides=(2, 2), padding='valid')
    inception = conv2d_bn(image_input, 32, 3, 3, strides=(1, 1), padding = 'valid')
    inception = conv2d_bn(inception, 32, 3, 3, padding='valid')
    inception = conv2d_bn(inception, 64, 3, 3)
    inception = MaxPooling2D((3, 3), strides=(2, 2))(inception)

    inception = conv2d_bn(inception, 80, 1, 1, padding='valid')
    inception = conv2d_bn(inception, 192, 3, 3, padding='valid')
    inception = MaxPooling2D((3, 3), strides=(2, 2))(inception)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(inception, 64, 1, 1)

    branch5x5 = conv2d_bn(inception, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(inception, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inception)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    inception = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis, name='mixed0')
    if mixed > 0:
         # mixed 1: 35 x 35 x 256
        branch1x1 = conv2d_bn(inception, 64, 1, 1)

        branch5x5 = conv2d_bn(inception, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(inception, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inception)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        inception = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis, name='mixed1')
    if mixed > 1:
        # mixed 2: 35 x 35 x 256
        branch1x1 = conv2d_bn(inception, 64, 1, 1)

        branch5x5 = conv2d_bn(inception, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(inception, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inception)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        inception = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis, name='mixed2')
    if mixed > 2:
        # mixed 3: 17 x 17 x 768
        # branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
        branch3x3 = conv2d_bn(inception, 384, 3, 3, strides=(1, 1), padding='valid')

        branch3x3dbl = conv2d_bn(inception, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(1, 1), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(1, 1))(inception)
        inception = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis, name='mixed3')
    if mixed > 3:
        # mixed 4: 17 x 17 x 768
        branch1x1 = conv2d_bn(inception, 192, 1, 1)

        branch7x7 = conv2d_bn(inception, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(inception, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inception)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        inception = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis, name='mixed4')
    if mixed > 4:
        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = conv2d_bn(inception, 192, 1, 1)

            branch7x7 = conv2d_bn(inception, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(inception, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(inception)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            inception = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis, name='mixed' + str(5 + i))
    if mixed > 6:
        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(inception, 192, 1, 1)

        branch7x7 = conv2d_bn(inception, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(inception, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inception)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        inception = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis, name='mixed7')
    if mixed > 7:
        # mixed 8: 8 x 8 x 1280
        branch3x3 = conv2d_bn(inception, 192, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                            strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn(inception, 192, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(inception)
        inception = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool],
            axis=channel_axis, name='mixed8')
    if mixed > 8:
        # mixed 9, 10: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = conv2d_bn(inception, 320, 1, 1)

            branch3x3 = conv2d_bn(inception, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2],
                axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(inception, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2],
                axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(inception)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            inception = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis, name='mixed' + str(9 + i)
    if include_top:
        # Classification block
        inception = GlobalAveragePooling2D(name='avg_pool')(inception)
        output = Dense(classes, activation="sigmoid", name="predictions")(inception)
    else:
        if pooling == 'avg':
            output = GlobalAveragePooling2D()(inception)
        elif pooling == 'max':
            output = GlobalMaxPooling2D()(inception)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = image_input
    # Create model.
    model = Model(inputs, output, name='inception_v3')
    # load weights
    if weights:
        model.load_weights(weights)

    return model

