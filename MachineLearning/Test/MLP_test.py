import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


# X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
# X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
#
# mlp=MLPClassifier(solver='lbfgs',activation='tanh',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
# mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
# mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
mlp=MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation='none',cmap='viridis')
plt.yticks(range(30),cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()