import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


iris=load_iris()
logreg=LogisticRegression()
scores=cross_val_score(logreg,iris.data,iris.target,cv=5)
print("Cross-validation scores:{}".format(scores))
kfold=KFold(n_splits=5)
print("Cross-validation scores:\n{}".format(cross_val_score(logreg,iris.data,iris.target,cv=kfold)))