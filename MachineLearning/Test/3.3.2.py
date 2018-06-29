from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=1)
# scaler=MinMaxScaler()

# scaler.fit(X_train)
# X_train_scaled=scaler.transform(X_train)
# X_test_scaled=scaler.transform(X_test)

# fig,axes=plt.subplots(15,2,figsize=(10,20))
# maliganant=cancer.data[cancer.target==0]
# benign=cancer.data[cancer.target==1]
#
# ax=axes.ravel()
#
# for i in range(30):
#     _,bins=np.histogram(cancer.data[:,i],bins=50)
#     ax[i].hist(maliganant[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
#     ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
#     ax[i].set_title(cancer.feature_names[i])
#     ax[i].set_yticks(())
# ax[0].set_xlabel("Feature magnitude")
# ax[0].set_ylabel("Frequency")
# ax[0].legend(["maglignant","benign"],loc="best")
# fig.tight_layout()
# fig.show()
#
# scaler=StandardScaler()
# scaler.fit(cancer.data)
# X_scaled=scaler.transform(cancer.data)
# pca=PCA(n_components=2)
# pca.fit(X_scaled)
# X_pca=pca.transform(X_scaled)
# print("Original shape:{}".format(str(X_scaled.shape)))
# print("Reduced shape:{}".format(str(X_pca.shape)))
#
# plt.figure(figsize=(8,8))
# mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
# plt.legend(cancer.target_names,loc="best")
# plt.gca().set_aspect("equal")
# plt.xlabel("First principal component")
# plt.ylabel("Second principal component")
# plt.show()

# people=fetch_lfw_people(min_faces_per_person=20,resize=0.7)
# image_shape=people.images[0].shape
#
# fix,axes=plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),"yticks":()})
# for target,image,ax in zip(people.target,people.images,axes.ravel()):
#     ax.imshow(image)
#     ax.set_title(people.target_names[target])
#
# print("people.images.shape:{}".format(people.images.shape))
# print("Number of classes:{}".format(len(people.target_names)))

# X,y=make_blobs(random_state=1)
# kmeans=KMeans(n_clusters=3)
# kmeans.fit(X)
# print(kmeans.predict(X))

X_train,X_test,y_train,y_test=train_test_split(X_people,y_people,stratify=y_people,random_state=0)
nmf=NMF(n_components=100,random_state=0)
pca=PCA(n_components=100,random_state=0)
kmeans=KMeans(n_cluster=100,random_state=0)
nmf.fit(X_train)
pca.fit(X_train)
kmeans.fit(X_train)

X_reconstructed_pca=pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans=kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf=np.dot(nmf.transform(X_test),nmf.components_)
