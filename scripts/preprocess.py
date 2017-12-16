#import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import os
import pandas as pd
from iceburger.io import (denoise, parse_json_data, smooth, grayscale,
                         image_normalization)
weight_smooth = 0.25
weight_denoise = 0.15
DATA = "/workspace/iceburger/data/processed"
f_p = "/workspace/iceburger/data/processed/train_valid_knn11_impute.json"
df = pd.read_json(f_p)
dim = int(np.sqrt(len(df.band_1.iloc[0])))
X, X_angle, y, subset = parse_json_data(f_p, padding="avg")
X_smooth_denoise = np.array([smooth(denoise(x, weight_denoise, True), weight_smooth) for x in X])
XX=X_smooth_denoise.reshape([len(X),dim*dim,3])
df.band_1 = [XX[i,:,0] for i in range(len(X))]
df.band_2 = [XX[i,:,1] for i in range(len(X))]
df.to_json(os.path.join(DATA,"train_valid_knn11_impute_denoise_smooth.json"))
"""
output = []
[output.append((df.id[i],X[i],X_angle[i],y[i],subset[i])) for i in range(len(X))]
df_output = pd.DataFrame.from_records(output)
df_output.columns = ['id', 'X', 'X_angle','Y', 'subset']
df_output.to_json(os.path.join(DATA, 'train_valid_knn11_impute_denoise_smooth.json'))
"""
