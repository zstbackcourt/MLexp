import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

# X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
# X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
#
# forest=RandomForestClassifier(n_estimators=5,random_state=2)
# forest.fit(X_train,y_train)
#
# fig,axes=plt.subplots(2,3,figsize=(20,10))
# for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
#     ax.set_title("Tree {}".format(i))
#     mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)
#
# mglearn.plots.plot_2d_separator(forest,X_train,fill=True,ax=axes[-1,-1],alpha=.4)
# axes[-1,-1].set_title("Random Forest")
# mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
# fig.show()
#
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
gbrt=GradientBoostingClassifier(random_state=0,max_depth=1,learning_rate=0.01)
gbrt.fit(X_train,y_train)
print("训练集的精度为:{}".format(gbrt.score(X_train,y_train)))
print("测试集的精度为:{}".format(gbrt.score(X_test,y_test)))
