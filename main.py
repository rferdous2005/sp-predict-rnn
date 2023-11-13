import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from process import *

scaled_df = collectDataForASymbol(2016, 2022, "KOHINOOR")
time_steps = 60
X_train = []
y_train = []

print(scaled_df.loc[59])
for i in range(60, len(scaled_df)):
    X_train.append(scaled_df.loc[i-60: i, "Close"])
    y_train.append(scaled_df.loc[i, "Close"])

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train, y_train)
