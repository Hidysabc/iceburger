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
from keras import layers
from keras.engine.topology import get_source_inputs
from keras.layers import (Input, Activation, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, GlobalAveragePooling2D,
                          AveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Permute,
                          Reshape)
from keras.constraints import non_neg
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from iceburger.io import parse_json_data,color_composite, smooth, denoise, grayscale

FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-combined-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)

train_all = True
# These are train flags that required to train model more efficiently and
# select proper model parameters
train_common = False
train_b = False or train_all
train_img = False or train_all
train_total = False or train_all
predict_submission = True and train_all

clean_all = True
clean_b = False or clean_all
clean_img = True or clean_all

load_all = True
load_b = False or load_all
load_img = False or load_all
"""
# Original parameters --- baseline
smooth_rgb=0.2
smooth_gray=0.5
weight_rgb=0.05
weight_gray=0.05
DROPOUT_fcnn=0.2
DROPOUT_combined=0.3
"""
DROPOUT_resnet=0.3
DROPOUT_combined=0.4

#: Regularization
L2R = 5E-3

def create_dataset(json_filename, labeled, smooth_rgb=0.2, smooth_gray=0.2,
                   weight_rgb=0.1, weight_gray=0.1):
    df = pd.read_json(json_filename)
    band_1, band_2, images = df['band_1'].values, df['band_2'].values, color_composite(df)
    to_arr = lambda x: np.asarray([np.asarray(item) for item in x])
    band_1 = to_arr(band_1)
    band_2 = to_arr(band_2)
    band_3 = (band_1 + band_2) / 2
    gray_reshape = lambda x: np.asarray([item.reshape(75, 75) for item in x])
    # Make a picture format from flat vector
    band_1 = gray_reshape(band_1)
    band_2 = gray_reshape(band_2)
    band_3 = gray_reshape(band_3)
    print('Denoising and reshaping')
    if train_b and clean_b:
        # Smooth and denoise data
        band_1 = smooth(denoise(band_1, weight_gray, False), smooth_gray)
        print('Gray 1 done')
        band_2 = smooth(denoise(band_2, weight_gray, False), smooth_gray)
        print('Gray 2 done')
        band_3 = smooth(denoise(band_3, weight_gray, False), smooth_gray)
        print('Gray 3 done')
    if train_img and clean_img:
        images = smooth(denoise(images, weight_rgb, True), smooth_rgb)
    print('RGB done')
    tf_reshape = lambda x: np.asarray([item.reshape(75, 75, 1) for item in x])
    band_1 = tf_reshape(band_1)
    band_2 = tf_reshape(band_2)
    band_3 = tf_reshape(band_3)
    #images = tf_reshape(images)
    band = np.concatenate([band_1, band_2, band_3], axis=3)
    if labeled:
        y = np.array(df["is_iceberg"])
    else:
        y = None
    return y, band, images

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
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
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
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


def get_model_notebook(lr, decay, channels, relu_type='relu'):
    # angle variable defines if we should use angle parameter or ignore it
    img_input = Input(shape=(75, 75, channels))
    resnet = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    resnet = BatchNormalization(axis=3, name='bn_conv1')(resnet)
    resnet = Activation(relu_type)(resnet)
    resnet = MaxPooling2D((3, 3), strides=(2, 2))(resnet)
    #resnet= Dropout(DROPOUT_resnet)(resnet)
    resnet = conv_block(resnet, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    resnet = identity_block(resnet, 3, [64, 64, 256], stage=2, block='b')
    resnet = identity_block(resnet, 3, [64, 64, 256], stage=2, block='c')
    #resnet= Dropout(DROPOUT_resnet)(resnet)
    """
    resnet = conv_block(resnet, 3, [128, 128, 512], stage=3, block='a')
    resnet = identity_block(resnet, 3, [128, 128, 512], stage=3, block='b')
    resnet = identity_block(resnet, 3, [128, 128, 512], stage=3, block='c')
    resnet = identity_block(resnet, 3, [128, 128, 512], stage=3, block='d')
    """
    #resnet = MaxPooling2D((2, 2), strides=(2, 2))(resnet)
    #resnet = Dropout(DROPOUT_resnet)(resnet)
    #resnet = BatchNormalization()(resnet)
    #resnet = AveragePooling2D((9,9), name = "avg_pool")(resnet)
    #resnet = Flatten()(resnet)
    resnet = GlobalMaxPooling2D()(resnet)
    #local_input = img_input
    partial_model = Model(img_input, resnet)
    LOG.info("Partial_model_summary:")
    partial_model.summary()
    dense = Dense(512, name="fc1")(resnet)
    dense = BatchNormalization(name = "bn_fc1")(dense)
    dense = Activation(relu_type)(dense)
    dense = Dropout(DROPOUT_resnet)(dense)
    #dense = Dropout(DROPOUT_resnet)(dense)
    dense = Dense(256, activation=relu_type)(dense)
    dense = Dropout(DROPOUT_resnet)(dense)
    dense = Dense(128, activation=relu_type)(dense)
    dense = Dropout(DROPOUT_resnet)(dense)
    #dense = Dense(64, activation=relu_type)(dense)
    #dense = Dropout(DROPOUT_resnet)(dense)
    output = Dense(1, activation="sigmoid", name="predictions")(dense)
    """
    fcnn = Conv2D(32, kernel_size=(3, 3), activation=relu_type)(
        BatchNormalization()(input_1))
    fcnn = MaxPooling2D((3, 3))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = Conv2D(64, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(DROPOUT_fcnn)(fcnn)
    fcnn = BatchNormalization()(fcnn)
    fcnn = Flatten()(fcnn)
    local_input = input_1
    partial_model = Model(input_1, fcnn)
    dense = Dropout(DROPOUT_fcnn)(fcnn)
    dense = Dense(256, activation=relu_type)(dense)
    dense = Dropout(DROPOUT_fcnn)(dense)
    dense = Dense(128, activation=relu_type)(dense)
    dense = Dropout(DROPOUT_fcnn)(dense)
    dense = Dense(64, activation=relu_type)(dense)
    dense = Dropout(DROPOUT_fcnn)(dense)
    # For some reason i've decided not to normalize angle data
    output = Dense(1, activation="sigmoid")(dense)
    """
    model = Model(img_input, output)
    LOG.info("model summary:")
    model.summary()
    #model = Model(local_input, output)
    #optimizer = Adam(lr=lr, decay=decay)
    optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, partial_model


def combined_model(m_b, m_img, lr, decay):
    input_b = Input(shape=(75, 75, 3))
    input_img = Input(shape=(75, 75, 3))

    # I've never tested non-trainable source models tho
    #for layer in m_b.layers:
    #    layer.trainable = False
    #for layer in m_img.layers:
    #    layer.trainable = False

    m1 = m_b(input_b)
    m2 = m_img(input_img)

    # So, combine models and train perceptron based on that
    # The iteresting idea is to use XGB for this task, but i actually hate this method
    common = Concatenate()([m1, m2])
    common = BatchNormalization()(common)
    common = Dropout(DROPOUT_combined)(common)
    common = Dense(1024, activation='relu')(common)
    common = Dropout(DROPOUT_combined)(common)
    common = Dense(512, activation='relu')(common)
    common = Dropout(DROPOUT_combined)(common)
    output = Dense(1, activation="sigmoid")(common)
    model = Model([input_b, input_img], output)
    #optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def gen_flow_multi_inputs(I1, I2, y, batch_size):
    gen1 = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             rotation_range=10)
    gen2 = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             rotation_range=10)
    genI1 = gen1.flow(I1, y, batch_size=batch_size, seed=57, shuffle=True)
    genI2 = gen2.flow(I1, I2, batch_size=batch_size, seed=57, shuffle=True)
    while True:
        I1i = genI1.next()
        I2i = genI2.next()
        #print I1i[0].shape
        np.testing.assert_array_equal(I2i[0], I1i[0])
        yield [I1i[0], I2i[1]], I1i[1]
"""
#Train a particular model
def train_model(model, lr, batch_size, epochs, checkpoint_name, X_train, y_train, val_data, verbose=2):
    callbacks = [ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss')]
    callbacks.append(
        LearningRateScheduler(lambda epoch: max(1e-5, lr * (0.75 ** (epoch // 5))))
    )
    datagen = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.,
                                   height_shift_range=0.,
                                   channel_shift_range=0,
                                   zoom_range=0.2,
                                   rotation_range=10)
    x_test, y_test = val_data
    try:
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True), epochs=epochs,
                                    steps_per_epoch=len(X_train) / batch_size,
                                    validation_data=(x_test, y_test), verbose=1,
                                    callbacks=callbacks)
    except KeyboardInterrupt:
        if verbose > 0:
            print('Interrupted')
    if verbose > 0:
        print('Loading model')
    model.load_weights(filepath=checkpoint_name)
    return model
"""


#Train a particular model
def train_model(model, lr, batch_size, epochs, checkpoint_name, X_train, y_train, val_data, verbose=2):
    if train_common:
        if verbose > 0:
            print('Loading model weights from: {}'.format(checkpoint_name))
        model.load_weights(checkpoint_name)
        return model
    else:
        callbacks = [ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss')]
        callbacks.append(
            LearningRateScheduler(lambda epoch: max(1e-4, lr * (0.75 ** (epoch // 5))))
        )
        datagen = ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True,
                                       width_shift_range=0.,
                                       height_shift_range=0.,
                                       channel_shift_range=0,
                                       zoom_range=0.2,
                                       rotation_range=10)
        x_test, y_test = val_data
        try:
            model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True), epochs=epochs,
                                        steps_per_epoch=len(X_train) / batch_size,
                                        validation_data=(x_test, y_test), verbose=1,
                                        callbacks=callbacks)
        except KeyboardInterrupt:
            if verbose > 0:
                print('Interrupted')
        if verbose > 0:
            print('Loading model')
        model.load_weights(filepath=checkpoint_name)
    return model

def gen_model_weights(lr, decay, channels, relu, batch_size, epochs, path_name, data, verbose=2):
    X_train, y_train, X_test, y_test, X_val, y_val = data
    model, partial_model = get_model_notebook(lr, decay, channels, relu)
    model = train_model(model, lr, batch_size, epochs, path_name,
                           X_train, y_train, (X_test, y_test), verbose=verbose)

    if verbose > 0:
        loss_val, acc_val = model.evaluate(X_val, y_val,
                               verbose=0, batch_size=batch_size)

        loss_train, acc_train = model.evaluate(X_test, y_test,
                                       verbose=0, batch_size=batch_size)

        print('Val/Train Loss:', str(loss_val) + '/' + str(loss_train), \
            'Val/Train Acc:', str(acc_val) + '/' + str(acc_train))
    return model, partial_model

# Train all 3 models
def train_models(args,dataset, lr, batch_size, max_epoch, verbose=2, return_model=False):
    y_train, X_b, X_images = dataset
    y_train_full, y_val,\
    X_b_full, X_b_val,\
    X_images_full, X_images_val = train_test_split(y_train, X_b, X_images, random_state=687, train_size=0.9)

    y_train_train, y_test, \
    X_b_train, X_b_test, \
    X_images_train, X_images_test = train_test_split(y_train_full, X_b_full, X_images_full, random_state=576, train_size=0.85)

    if train_b or train_common:
        if verbose > 0:
            print('Training bandwidth network')
        data_b1 = (X_b_train, y_train_train, X_b_test, y_test, X_b_val, y_val)
        LOG.info("Create callback functions")
        model_out_path = os.path.join(os.path.abspath(args.outpath),"checkpoints")
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        checkpoint_name= "{mn}-resnet-best_val_loss.hdf5".format(mn="model_b")
        model_b_outpath = os.path.join(model_out_path, checkpoint_name)
        model_b, model_b_cut = gen_model_weights(lr, 1e-6, 3, 'relu', batch_size, max_epoch, model_b_outpath,
                                                 data=data_b1, verbose=verbose)

    if train_img or train_common:
        if verbose > 0:
            print('Training image network')
        data_images = (X_images_train, y_train_train, X_images_test, y_test, X_images_val, y_val)
        LOG.info("Create callback functions")
        model_out_path = os.path.join(os.path.abspath(args.outpath),"checkpoints")
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        checkpoint_name= "{mn}-resnet-best_val_loss.hdf5".format(mn="model_img")
        model_img_outpath = os.path.join(model_out_path, checkpoint_name)
        model_images, model_images_cut = gen_model_weights(lr, 1e-6, 3, 'relu', batch_size, max_epoch, model_img_outpath,
                                                       data_images, verbose=verbose)

    if train_total or train_common:
        common_model = combined_model(model_b_cut, model_images_cut, lr/6, 1e-7)
        common_x_train = [X_b_full, X_images_full]
        common_y_train = y_train_full
        common_x_val = [X_b_val, X_images_val]
        common_y_val = y_val
        LOG.info("Create callback functions")
        model_out_path = os.path.join(os.path.abspath(args.outpath),"checkpoints")
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        checkpoint_name= "{mn}-resnet-best_val_loss.hdf5".format(mn="model_common")
        model_common_outpath = os.path.join(model_out_path, checkpoint_name)
        if verbose > 0:
            print('Training common network')
        callbacks = [ModelCheckpoint(model_common_outpath, save_best_only=True, monitor='val_loss')]
        callbacks.append(
            LearningRateScheduler(lambda epoch: max(1e-4, lr/6 * (0.75 ** (epoch // 5))))

        )
        try:
            common_model.fit_generator(gen_flow_multi_inputs(X_b_full, X_images_full, y_train_full, batch_size),
                                         epochs=max_epoch,
                                  steps_per_epoch=len(X_b_full) / batch_size,
                                  validation_data=(common_x_val, common_y_val), verbose=1,
                                  callbacks=callbacks)
        except KeyboardInterrupt:
            pass
        common_model.load_weights(filepath=model_common_outpath)
        loss_val, acc_val = common_model.evaluate(common_x_val, common_y_val,
                                           verbose=0, batch_size=batch_size)
        loss_train, acc_train = common_model.evaluate(common_x_train, common_y_train,
                                                  verbose=0, batch_size=batch_size)
        if verbose > 0:
            print('Loss:', loss_val, 'Acc:', acc_val)
    if return_model:
        return common_model
    else:
        return (loss_train, acc_train), (loss_val, acc_val)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data", type=str, metavar="DATA",
        help=("Path to training data stored in json."))
    parser.add_argument(
        "--model", type=str, metavar="MODEL", default= "combined_model",
        help="Model type for training (Options: combined_model)")
    parser.add_argument(
        "--batch_size", type=int, metavar="BATCH_SIZE", default=32,
        help="Number of samples in a mini-batch")
    parser.add_argument(
        "--epochs", type=int, metavar="EPOCHS", default=100,
        help="Number of epochs")
    parser.add_argument(
        "--outpath", type=str, metavar="OUTPATH",
        default="./",
        help="Output path where parsed data set class to be saved")
    args = parser.parse_args()
    y_train, X_b, X_images = create_dataset(args.data, True)
    # baseline parameters
    #common_model = train_models(args,(y_train, X_b, X_images), 7e-04, args.batch_size, 50, 1, return_model=True)
    common_model = train_models(args,(y_train, X_b, X_images), 5e-03, args.batch_size, args.epochs, 1, return_model=True)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
