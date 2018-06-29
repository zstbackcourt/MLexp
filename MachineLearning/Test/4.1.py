import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv(
    "D:\\PyCharm\\workplace\\MachineLearning\\Test\\adult.data",header=None,index_col=False,
    names=[
        'age','workclass','fnlwgt','education','education-num',
        'marital-status','occupation','relationship','race','gender',
        'capital-gain','capital-loss','hours-per-week','native-country',
        'income'
    ]
)

data=data[['age','workclass','education','gender','hours-per-week','occupation','income']]
display(data.head())

print(data.gender.value_counts())
data_dummies=pd.get_dummies(data)
data_dummies.head()
print(data_dummies.head())

features=data_dummies.ix[:,'age':'occupation_ Transport-moving']
X=features.values
y=data_dummies['income_ >50K'].values
print("X,shape:{},y.shape:{}".format(X.shape,y.shape))

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
print("Test score:{:.2f}".format(logreg.score(X_test,y_test)))

