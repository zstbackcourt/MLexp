import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import ElasticNet

# X,y=mglearn.datasets.make_wave(n_samples=60)
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
# lr=LinearRegression().fit(X_train,y_train)
# print("斜率:{}".format(lr.coef_))
# print("截距:{}".format(lr.intercept_))

#岭回归
# X,y=mglearn.datasets.load_extended_boston()
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
# ridge=Ridge().fit(X_train,y_train)
# print("Training set score:{:.2f}".format(ridge.score(X_train,y_train)))
# print("Test set score:{:.2f}".format(ridge.score(X_test,y_test)))

#lasson回归
# X,y=mglearn.datasets.load_extended_boston()
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
# lasso=Lasso().fit(X_train,y_train)
# lasso001=Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
# lasso00001=Lasso(alpha=0.0001,max_iter=100000).fit(X_train,y_train)
# ridge01=Ridge(alpha=0.1).fit(X_train,y_train)
# print("Training set score:{:.2f}".format(lasso001.score(X_train,y_train)))
# print("Test set score:{:.2f}".format(lasso001.score(X_test,y_test)))
# print("Number of features used:{}".format(np.sum(lasso001.coef_!=0)))
#
# plt.plot(lasso.coef_,'s',label="Lasso alpha=1")
# plt.plot(lasso001.coef_,'^',label="Lasso alpha=0.01")
# plt.plot(lasso00001.coef_,'v',label="Lasso alpha=0.0001")
# plt.plot(ridge01.coef_,'o',label="Ridge alpha=0.1")
# plt.legend(ncol=1,loc=(0,1.05))
# plt.ylim(-25,25)
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.show()


#Logistic回归
# X,y=mglearn.datasets.make_forge()
#
# fig ,axes=plt.subplots(1,2,figsize=(10,3))
#
# for model,ax in zip([LinearSVC(),LogisticRegression()],axes):
#     clf=model.fit(X,y)
#     mglearn.plots.plot_2d_separator(clf,X,fill=False,eps=0.5,ax=ax,alpha=.7)
#     mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# axes[0].legend()
# fig.show()

#Logistic回归乳腺癌
# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
# logreg=LogisticRegression().fit(X_train,y_train)
# logreg100=LogisticRegression(C=100).fit(X_train,y_train)
# logreg001=LogisticRegression(C=0.01).fit(X_train,y_train)
#      .T 的作用是将矩阵转置
# plt.plot(logreg.coef_.T,'o',label="C=1")
# plt.plot(logreg100.coef_.T,'^',label="C=100")
# plt.plot(logreg001.coef_.T,'v',label="C=0.01")
# plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
# plt.hlines(0,0,cancer.data.shape[1])
# plt.legend()
# plt.ylim(-5,5)
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.show()

#OvR
# X,y=make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(["Class 0","Class 1","Class 2"])
#plt.show()

X,y=make_blobs(random_state=42)
linear_svm=LinearSVC().fit(X,y)
print("Coefficient shape:",linear_svm.coef_.shape)
print("Intercept shape:",linear_svm.intercept_.shape)
#mglearn.plots.plot_2d_classification(linear_svm,X,fill=True,alpha=.7)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
line=np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2","Line Class 0","Line Class 1","Line Class 2"],loc=(1.01,0.3))
plt.show()