import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from mglearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D,axes3d
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


#
# X,y=make_blobs(centers=4,random_state=8)
# y=y%2
# linear_svm=LinearSVC().fit(X,y)
# mglearn.plots.plot_2d_separator(linear_svm,X)
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
#
#
# X_new=np.hstack([X,X[:,1:]**2])
# figure=plt.figure()
# ax=Axes3D(figure,elev=-152,azim=-26)
# mask=y==0
# ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
# ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker='^',cmap=mglearn.cm2,s=60)
# ax.set_xlabel("Feature 0")
# ax.set_ylabel("Feature 1")
# ax.set_zlabel("Feature 1 ** 2")
# figure.show()
#
# X,y=mglearn.tools.make_handcrafted_dataset()
# svm=SVC(kernel='rbf',C=10,gamma=0.1).fit(X,y)
# mglearn.plots.plot_2d_separator(svm,X,eps=.5)
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
#
# sv=svm.support_vectors_
# sv_labels=svm.dual_coef_.ravel()>0
# mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth=3)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()
#
# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
#
# svc=SVC()
# svc.fit(X_train,y_train)