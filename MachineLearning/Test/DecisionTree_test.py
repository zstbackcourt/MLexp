import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from sklearn.linear_model import LinearRegression


#p58
# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
# tree=DecisionTreeClassifier(max_depth=4, random_state=0)
# tree.fit(X_train,y_train)
# print("训练精度:{:.3f}".format(tree.score(X_train,y_train)))
# print("测试精度:{:.3f}".format(tree.score(X_test,y_test)))
# print("特征的重要性:\n{}".format(tree.feature_importances_))
# def plot_feature_importances_cancer(model):
#     n_features=cancer.data.shape[1]
#     plt.barh(range(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),cancer.feature_names)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
#     plt.show()
# plot_feature_importances_cancer(tree)

# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
# tree=DecisionTreeClassifier(max_depth=4, random_state=0)
# export_graphviz(tree,out_file="tree.dot",class_names=["malignant","benign"],feature_names=cancer.feature_names,impurity=False,filled=True)
# with open("tree.dot") as f:
#     dot_graph=f.read()
# graphviz.Source(dot_graph)

#p62
ram_prices=pd.read_csv("ram_price.csv")
data_train=ram_prices[ram_prices.date<2000]
data_test=ram_prices[ram_prices.date>=2000]
X_train=data_train.date[:,np.newaxis]
y_train=np.log(data_train.price)

tree=DecisionTreeRegressor().fit(X_train,y_train)
linear_reg=LinearRegression().fit(X_train,y_train)

X_all=ram_prices.date[:,np.newaxis]
pred_tree=tree.predict(X_all)
pred_lr=linear_reg.predict(X_all)

price_tree=np.exp(pred_tree)
price_lr=np.exp(pred_lr)

plt.semilogy(data_train.date,data_train.price,label="Training data")
plt.semilogy(data_test.date,data_test.price,label="Test data")
plt.semilogy(ram_prices.date,price_tree,label="Tree prediction")
plt.semilogy(ram_prices.date,price_lr,label="Linear prediction")
plt.legend()
plt.show()
