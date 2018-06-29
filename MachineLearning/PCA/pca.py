from numpy import *

"""载入数据"""
def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]  #将数据存储在一个数组中
    datArr=[list(map(float,line)) for line in stringArr]   #将数据转化为float的映射
    return mat(datArr)


def pca(dataMat,topNfeat=9999999):
    meanVals=mean(dataMat,axis=0)      #计算每列的平均值，也就是所有样本每个特征的平均值
    meanRemoved=dataMat-meanVals       #中心化，即每个样本的特征减去该特征的平均值
    covMat=cov(meanRemoved,rowvar=0)   #计算中心化矩阵的协方差
    eigVals,eigVects=linalg.eig(mat(covMat))   #返回特征值和特征向量
    eigValInd=argsort(eigVals)               #返回特征值数组中从小到大的排序（注意返回的是索引值，而不是实际的数值）
    eigValInd=eigValInd[:-(topNfeat+1):-1]   #将数组倒序
    redEigVects=eigVects[:,eigValInd]        #得到对应的特征向量
    lowDDataMat=meanRemoved*redEigVects      #得到新的低维的数据集
    reconMat=(lowDDataMat*redEigVects.T)+meanVals     #降维后的数据集重构
    return lowDDataMat,reconMat



if __name__=='__main__':
    dataMat=loadDataSet('testSet.txt')
    lowDMat,reconMat=pca(dataMat,4)
    print(shape(lowDMat))