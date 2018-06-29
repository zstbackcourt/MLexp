from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split

X,y=make_circles(noise=0.25,factor=0.5,random_state=1)
y_named=np.array(["blue","red"])[y]
X_train,X_test,y_train_named,y_test_named,y_train,y_test=train_test_split(X,y_named,y,random_state=0)


gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train_named)

print("X_test.shape:{}".format(X_test.shape))
print("Decision function shape:{}".format(gbrt.decision_function(X_test).shape))