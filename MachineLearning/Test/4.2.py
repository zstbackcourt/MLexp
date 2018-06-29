import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

X,y=mglearn.datasets.make_wave(n_samples=100)


bins=np.linspace(-3,3,11)
which_bin=np.digitize(X,bins=bins)
encoder=OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned=encoder.transform(which_bin)
print(X_binned[:5])