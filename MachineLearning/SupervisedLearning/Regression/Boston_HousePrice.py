from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

boston=load_boston()
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

def LinearRegression_test():
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    lr_y_predict=lr.predict(X_test)
    print("简单线性回归模型评价：{}".format(lr.score(X_test,y_test)))
    print("使用R-squared评价标准：{}".format(r2_score(y_test,lr_y_predict)))
    print("使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))))
    print("使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))))

def SGDRegressor_test():
    sgdr=SGDRegressor(max_iter=1000)
    sgdr.fit(X_train, y_train.ravel())
    sgdr_y_predict = sgdr.predict(X_test)
    print("快速的随机梯度下降模型评价：{}".format(sgdr.score(X_test, y_test)))
    print("使用R-squared评价标准：{}".format(r2_score(y_test, sgdr_y_predict)))
    print("使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))))
    print("使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))))

def SupportVectorRegression_test():
    linear_svr=SVR(kernel='linear')
    linear_svr.fit(X_train,y_train.ravel())
    linear_svr_y_predict=linear_svr.predict(X_test)

    poly_svr=SVR(kernel='poly')
    poly_svr.fit(X_train,y_train.ravel())
    poly_svr_y_predict=poly_svr.predict(X_test)

    rbf_svr=SVR(kernel='rbf')
    rbf_svr.fit(X_train,y_train.ravel())#ravel()将2D改成1D
    rbf_svr_y_predict=rbf_svr.predict(X_test)

    print("对线性核函数使用R-squared评价标准：{}".format(linear_svr.score(X_test,y_test)))
    print("对线性核函数使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))))
    print("对线性核函数使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))))
    print("\n")
    print("对多项式核函数使用R-squared评价标准：{}".format(poly_svr.score(X_test, y_test)))
    print("对多项式核函数使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))))
    print("对多项式核函数使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))))
    print("\n")
    print("对径向基核函数使用R-squared评价标准：{}".format(rbf_svr.score(X_test, y_test)))
    print("对径向基核函数使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))))
    print("对径向基核函数使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))))

def KNeighborsRegressor_test():
    uni_knr=KNeighborsRegressor(weights='uniform')
    uni_knr.fit(X_train,y_train)
    uni_knr_y_predict=uni_knr.predict(X_test)

    dis_knr = KNeighborsRegressor(weights='distance')
    dis_knr.fit(X_train, y_train)
    dis_knr_y_predict = dis_knr.predict(X_test)

    print("对平均回归K近邻使用R-squared评价标准：{}".format(uni_knr.score(X_test, y_test)))
    print("对平均回归K近邻使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))))
    print("对平均回归K近邻使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))))
    print("\n")
    print("对距离加权回归K近邻使用R-squared评价标准：{}".format(dis_knr.score(X_test, y_test)))
    print("对距离加权回归K近邻使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))))
    print("对距离加权回归K近邻使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))))

def DecisionTreeRegressor_test():
    dtr=DecisionTreeRegressor()
    dtr.fit(X_train,y_train)
    dtr_y_predict=dtr.predict(X_test)
    print("对回归树使用R-squared评价标准：{}".format(dtr.score(X_test, y_test)))
    print("对回归树使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict))))
    print("对回归树使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict))))

def Ensemble_test():
    rfr=RandomForestRegressor()
    rfr.fit(X_train,y_train.ravel())
    rfr_y_predict=rfr.predict(X_test)

    etr=ExtraTreesRegressor()
    etr.fit(X_train,y_train.ravel())
    etr_y_predict=etr.predict(X_test)

    gbr=GradientBoostingRegressor()
    gbr.fit(X_train,y_train.ravel())
    gbr_y_predict=gbr.predict(X_test)

    print("对普通随机森林使用R-squared评价标准：{}".format(rfr.score(X_test, y_test)))
    print("对普通随机森林使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))))
    print("对普通随机森林使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))))
    print("\n")
    print("对极端回归森林使用R-squared评价标准：{}".format(etr.score(X_test, y_test)))
    print("对极端回归森林使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict))))
    print("对极端回归森林使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict))))
    print("极端回归森林模型中每种特征对预测目标的贡献度：")
    print(np.sort(list(zip(etr.feature_importances_,boston.feature_names)),axis=0))
    print("\n")
    print("对梯度提升回归树使用R-squared评价标准：{}".format(gbr.score(X_test, y_test)))
    print("对梯度提升回归树使用MAE评价标准：{}".format(mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict))))
    print("对梯度提升回归树使用MSE评价标准：{}".format(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict))))

if __name__=='__main__':
    print("波士顿房价预测")
    print("简单线性回归：1\n"
          "快速的随机梯度下降：2\n"
          "支持向量机回归模型：3\n"
          "K邻近回归模型：4\n"
          "回归树模型：5\n"
          "集成模型：6\n")
    No = input("请输入你要查看的算法：")
    No = int(No)
    if No == 1:
        LinearRegression_test()
    elif No==2:
        SGDRegressor_test()
    elif No==3:
        SupportVectorRegression_test()
    elif No==4:
        KNeighborsRegressor_test()
    elif No==5:
        DecisionTreeRegressor_test()
    elif No==6:
        Ensemble_test()

