import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X=titanic[['pclass','age','sex']]
y=titanic['survived']
X['age'].fillna(X['age'].mean(),inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
vec=DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

def DecisionTree_test():
    dtc=DecisionTreeClassifier()
    dtc.fit(X_train,y_train)
    y_predict=dtc.predict(X_test)
    print("简单决策树的准确性评价:{}".format(dtc.score(X_test,y_test)))
    print(classification_report(y_predict,y_test,target_names=['died','survived']))

def RandomForestClassifier_test():
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    print("随机森林的准确性评价:{}".format(rfc.score(X_test, y_test)))
    print(classification_report(y_predict, y_test, target_names=['died', 'survived']))

def GradientBoostingClassifier_test():
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_predict = gbc.predict(X_test)
    print("梯度提升决策树的准确性评价:{}".format(gbc.score(X_test, y_test)))
    print(classification_report(y_predict, y_test, target_names=['died', 'survived']))


if __name__=='__main__':
    print("简单决策树：1，随机森林：2，梯度提升决策树：3\n")
    No=input("请输入你要查看的算法：")
    No=int(No)
    if No==1:
        DecisionTree_test()
    elif No==2:
        RandomForestClassifier_test()
    else:
        GradientBoostingClassifier_test()