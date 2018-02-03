"""
Classes and functions related to ensemble keras models
"""


import copy
import glob
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


KFOLD_CONF_KEYS = ["n_splits", "n_repeats", "random_state"]
OPTIMIZER_CONF_KEYS = ["optimizer_class", "lr", "decay"]
COMPILE_CONF_KEYS = ["loss", "metrics"]
FIT_GENERATOR_CONF_KEYS = ["epochs", "callbacks"]


class KFoldEnsembleKerasModel(object):
    def __init__(self, model_arch_path, **kwargs):
        self.model_arch_path = model_arch_path
        splitter_config = {"n_splits": 2, "n_repeats": 1}
        for conf in KFOLD_CONF_KEYS:
            if conf in kwargs:
                splitter_config[conf] = kwargs[conf]

        if splitter_config["n_repeats"] == 1:
            splitter_config.pop("n_repeats")
            self.kfold_splitter = StratifiedKFold(**splitter_config)
        else:
            self.kfold_splitter = RepeatedStratifiedKFold(**splitter_config)

        self.kfold_info = {}
        self.models = {}

    def train_generator(self, X, X_angle, y, gen, batch_size=32,
                        checkpoint_name="kfold-keras-model",
                        **kwargs):
        split = 0

        fitgen_config = {}
        for conf in FIT_GENERATOR_CONF_KEYS:
            if conf in kwargs:
                fitgen_config[conf] = kwargs[conf]

        for idx_train, idx_valid in self.kfold_splitter.split(X, y):
            split_name = "{0:03d}".format(split)
            _checkpoint_name = checkpoint_name + "_{}.hdf5".format(split_name)
            self.kfold_info[split_name] = {"idx_train": idx_train,
                                           "idx_valid": idx_valid,
                                           "checkpoint_path": _checkpoint_name}
            with open(self.model_arch_path, "r") as jsonfile:
                _model = model_from_json(jsonfile.read())

            optimizer_config = {"optimizer_class": SGD, "lr": 1e-3, "decay": 0}
            for conf in OPTIMIZER_CONF_KEYS:
                if conf in kwargs:
                    optimizer_config[conf] = kwargs[conf]

            compile_config = {}
            for conf in COMPILE_CONF_KEYS:
                if conf in kwargs:
                    compile_config[conf] = kwargs[conf]

            compile_config["optimizer"] = optimizer_config["optimizer_class"](
                lr=optimizer_config["lr"], decay=optimizer_config["decay"]
            )

            _model.compile(**compile_config)
            X_train = X[idx_train]
            X_angle_train = X_angle[idx_train]
            y_train = y[idx_train]
            X_valid = X[idx_valid]
            X_angle_valid = X_angle[idx_valid]
            y_valid = y[idx_valid]

            _fitgen_config = copy.copy(fitgen_config)
            _callbacks = [x for x in _fitgen_config["callbacks"]
                          if not isinstance(x, ModelCheckpoint)]
            _callbacks.append(ModelCheckpoint(_checkpoint_name,
                                              save_best_only=True,
                                              monitor="val_loss"))
            _fitgen_config["callbacks"] = _callbacks

            def gen_flow_for_two_input(X1, X2, y):
                genX1 = gen.flow(X1, y, batch_size=batch_size, shuffle=True)
                genX2 = gen.flow(X1, X2, batch_size=batch_size, shuffle=True)
                while True:
                    X1i = genX1.next()
                    X2i = genX2.next()
                    yield [X1i[0], X2i[1]], X1i[1]

            gen_train_ = gen_flow_for_two_input(X_train, X_angle_train, y_train)
            history = _model.fit_generator(
                gen_train_,
                validation_data=([X_valid, X_angle_valid], y_valid),
                steps_per_epoch=X_train.shape[0] / batch_size,
                **_fitgen_config)
            _model.load_weights(filepath=_checkpoint_name)

            val_loss = history.history["val_loss"]
            self.kfold_info[split_name]["history"] = history
            self.kfold_info[split_name]["best_epoch"] = np.argmin(val_loss)
            self.kfold_info[split_name]["best_val_loss"] = np.min(val_loss)
            self.models[split_name] = _model
            split += 1

        return self.kfold_info

    def load_weights(self, checkpoint_name):
        checkpoint_paths = glob.glob(checkpoint_name + "*.hdf5")
        for checkpoint_name in checkpoint_paths:
            split_name = checkpoint_name.split("_")[-1].replace(".hdf5", "")
            with open(self.model_arch_path, "r") as jsonfile:
                self.models[split_name] = model_from_json(jsonfile.read())

            self.models[split_name].load_weights(checkpoint_name)

    def predict(self, X, merge_function=(lambda x: np.mean(x, axis=1)),
                output_allfolds=False, **kwargs):
        y_pred = np.concatenate([self.models[n].predict(X, **kwargs)
                                 for n in self.models], axis=1)
        if output_allfolds:
            return y_pred
        else:
            return merge_function(y_pred)
