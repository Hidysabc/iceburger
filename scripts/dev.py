from iceburger.composite_model import CompositeModel
import json

#model_configs = json.load(open("convnet_resnet.json", "r"))
model_configs = json.load(open("convnet_convnet_denoise_smooth.json", "r"))
input_shape = (75, 75, 3)
m = CompositeModel(model_configs, input_shape)

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
