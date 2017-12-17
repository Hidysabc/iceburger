import os
from iceburger.io import parse_json_data
from iceburger.composite_model import CompositeModel
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import json

args = {}
args["data"] = "/tmp/iceburger/data/processed/train_valid_knn11_impute.json"

#model_configs = json.load(open("convnet_resnet.json", "r"))
model_configs = json.load(open("convnet_resnet.json", "r"))
input_shape = (75, 75, 3)
X, X_angle, y, subset = parse_json_data(os.path.join(args["data"]))
X_train = X[subset=='train']
X_angle_train = X_angle[subset=='train']
y_train = y[subset=='train']
X_valid = X[subset=='valid']
X_angle_valid = X_angle[subset=='valid']
y_valid = y[subset=='valid']

gen = ImageDataGenerator(horizontal_flip = True,
                    vertical_flip = True,
                     width_shift_range = 0.1,
                     height_shift_range = 0.1,
                     channel_shift_range = 0,
                     zoom_range = 0.2,
                     rotation_range = 30)
m = load_model(model_configs["resnet"]["path"])
m.compile(optimizer=m.optimizer, loss="binary_crossentropy", metrics=["accuracy"])
gen_train_ = gen.flow(X_train, y_train, batch_size = 32, seed=666)
gen_valid_ = gen.flow(X_valid, y_valid, batch_size = 32, seed=666)
history = m.fit_generator(
    gen_train_,
    steps_per_epoch=100,
    epochs=100, verbose=1,
    validation_data=gen_valid_,
    validation_steps=10)






from keras.models import load_model, Model
from keras.layers import Input, Concatenate

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



