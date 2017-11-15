'''
Training related functions and entry point
'''
import numpy as np
import pandas as pd
import cv2
import os
import sys
import argparse
import logging
import keras
import shutil

from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalMaxPooling2D,Dense,BatchNormalization,GlobalAveragePooling2D,Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from iceburger.io import parse_json_data

FORMAT =  '%(asctime)-15s %(name)-8s %(levelname)s %(message)s'
LOGNAME = 'iceburger-resnet50-train'

logging.basicConfig(format=FORMAT)
LOG = logging.getLogger(LOGNAME)
LOG.setLevel(logging.DEBUG)

PRJ = "/iceburger"
DATA = os.path.join(PRJ, "data/processed")



def get_callbacks(args,model_out_path):
    """
    Create list of callback functions for fitting step
    :param args: arguments as parsed by argparse module
    :returns: `list` of `keras.callbacks` classes
    """
    checkpoint_name= "{mn}-best_val_loss_weights.hdf5".format(mn="fcn")
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
            # stop training earlier if the model is not improving
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1, mode='auto'
        )
    )
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-8
        )
    )

    return callbacks, checkpoint_name


def compile_model(args, input_shape):
    """Build and compile model

    :param args: arguments as parsed by argparse module
    :returns: `keras.models.Model` of compiled model
    """
    base_model = ResNet50(include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = SGD(lr = 0.0001, momentum = 0.9)
    #for layer in base_model.layers:
    #    layer.trainable = False
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train(args):
    """Train a neural network

    :param args: arguments as parsed by argparse module
    """
    X, X_angle, y, subset = parse_json_data(os.path.join(DATA, "train_valid.json"))
    w = 197
    h = 197
    X_train = np.array([cv2.resize(x,(w,h)) for x in X[subset=='train']])
    X_angle_train = X_angle[subset=='train']
    y_train = y[subset=='train']
    X_valid = np.array([cv2.resize(x,(w,h)) for x in X[subset=='valid']])
    X_angle_valid = X_angle[subset=='valid']
    y_valid = y[subset=='valid']
    #ds = DataSet.from_pickle(args.data)
    #nb_classes = ds.df.activity.nunique()

    LOG.info("Create generators")
    #input_length = pd.Timedelta(args.window_time_s, "s")

    LOG.info("Initiate model")
    model = compile_model(args, input_shape=(197,197,3))

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
    def gen_flow_train_for_one_input(X1, y):
        genX1 = gen_train.flow(X1, y, batch_size= args.batch_size, seed=666)
        while True:
            X1i = genX1.next()
            yield X1i[0], X1i[1]

    def gen_flow_valid_for_one_input(X1, y):
        genX1 = gen_valid.flow(X1, y, batch_size= args.batch_size, seed=444)
        while True:
            X1i = genX1.next()
            yield X1i[0], X1i[1]
    #Finally create out generator
    #gen_train_ = gen_flow_train_for_one_input(X_train, y_train)
    #gen_valid_ = gen_flow_valid_for_one_input(X_valid, y_valid)
    gen_train_ = gen_train.flow(X_train, y_train)
    gen_valid_ = gen_valid.flow(X_valid, y_valid)

    """
    gen_train = ds.get_generator(batch_size=args.batch_size,subset="train",
                                 input_length=input_length,
                                 feature_cols=feature_cols,
                                 augmentations=augmentation_params)
    gen_valid = ds.get_generator(batch_size=args.batch_size, subset="valid",
                                 input_length=input_length,
                                 feature_cols=feature_cols,
                                 augmentations=augmentation_params)
    """

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
        mn="fcn",
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
        mn="fcn",
        val_loss=history.history["val_loss"][-1],
        val_acc=history.history["val_acc"][-1]
    )
    model.save(os.path.join(args.outpath, final_file_root))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        "--outpath", type=str, metavar="OUTPATH",
        default="./",
        help="Output path where parsed data set class to be saved")
    args = parser.parse_args()

    model = train(args)

    LOG.info("Done :)")


if __name__ == "__main__":
    sys.exit(main())
