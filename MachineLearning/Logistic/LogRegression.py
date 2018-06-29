from numpy import *
import matplotlib.pyplot as plt
import random

"""自定义训练数据集"""
def loadDataSet():
    dataMat=[]   #矩阵[x0,x1,x2],x0是b对应的x，值恒为1
    labelMat=[]  #标签向量
    fr=open("testSet.txt")
    for line in fr.readlines():
        lineArr=line.strip().split()  #strip()移除字符串头尾指定字符，默认空格;split()通过指定分隔符将字符串切片，默认为空格
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  #因为文件中读入的是字符串，所以将字符串转化为float类型
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

"""定义sigmoid函数"""
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# """梯度上升法"""
# def gradAscent(dataMatIn,classLabels):
#     dataMatrix=mat(dataMatIn)  #用dataMatIn创建特征矩阵
#     labelMat=mat(classLabels).transpose()  #调换矩阵的坐标顺序，对于二维矩阵来说，transpose()就是转置
#     m,n=shape(dataMatrix)  #m是样本数，n是特征数
#     alpha=0.001    #梯度上升步长
#     maxCycles=500  #最大迭代次数
#     weights=ones((n,1)) #权重向量b，初始化为全1
#     for k in range(maxCycles):
#         h=sigmoid(dataMatrix*weights)  #对w1*x1+w2*x2求对数几率回归
#         error=(labelMat-h)  #预测值和真实标签之间的误差
#         weights=weights+alpha*dataMatrix.transpose()*error  #梯度上升更新权重
#     return weights


"""画出数据集和Logistic回归的最佳拟合线的函数"""
def  plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)  #将矩阵转化为数组
    n=shape(dataMat)[0]     #n为样本数
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):      #遍历所有样本
        if int(labelMat[i])== 1:    #如果是正样本
            xcord1.append(dataArr[i,1])   #将正样本的x1属性的值加入xcord1
            ycord1.append(dataArr[i,2])   #将正样本的y1属性的值加入ycord1
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)   #画子图，参数一：子图的总行数；参数二：子图的总列数；参数三：子图的位置
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')   #绘制散点图
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]   #设置sigmoid函数为0，求解X2和X1的关系式，X0=1恒成立
    ax.plot(x,y)  #画线函数
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

"""随机梯度上升"""
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)  #m是样本数，n是特征数
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))    #随机梯度上升
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

"""改进的梯度上升算法"""
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)  #m是样本数，n是特征数
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[i]*weights))    #随机梯度上升
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights




"""测试"""
if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    weights=stocGradAscent1(array(dataArr),labelMat,500)
    plotBestFit(weights)