from numpy import *


def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))   #会根据提供的函数（类型）对指定序列做映射,python3中map返回的是迭代器，所以要改成list
        dataMat.append(fltLine)
    return dataMat


"""计算两个样本之间的距离"""
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

"""为给定的数据集构建一个包含k个随机簇质心的集合"""
"""随机质心必须在整个数据集的边界之内，通过找到数据集每一维的最小和最大值来完成"""
def randCent(dataSet,k):
    n=shape(dataSet)[1]             #特征数
    centroids=mat(zeros((k,n)))     #构建k个质心的矩阵，初始化为全0
    for j in range(n):
        minJ=min(dataSet[:,j])      #找出所有样本中每一个特征的最小特征的样本
        rangeJ=float(max(dataSet[:,j])-minJ)    #质心的特征可以设置的范围
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)    #随机k个质心
    return centroids

"""K-Means聚类"""
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]         #样本总数
    clusterAssment=mat(zeros((m,2)))    #创建一个矩阵，用于存储每个点的簇分配结果，包含两列：记录簇索引值；存储误差（当前点到簇质心的距离）
    centroids=createCent(dataSet,k)     #初始化簇质心
    clusterChanged=True                 #簇分配发生改变（因为簇质心发生了改变）
    while clusterChanged:               #迭代
        clusterChanged=False
        for i in range(m):              #遍历每一个样本
            minDist=inf
            minIndex=-1
            for j in range(k):           #用该样本和每一个当前质心比较
                distJI=distMeas(centroids[j,:],dataSet[i,:])    #找到距离最近的质心
                if distJI<minDist:
                    minDist=distJI        #记录最短距离
                    minIndex=j            #记录簇序号（索引）
            if clusterAssment[i,0]!=minIndex:     #查看当前样本的簇是否和现在分配的簇序号一致
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2    #记录误差
        print(centroids)
        for cent in range(k):    #更新质心
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)      #更新方式是计算簇中所有点的均值作为新的质心，axis=0表示沿矩阵列进行计算
    return centroids,clusterAssment




if __name__=='__main__':
    datMat=mat(loadDataSet('testSet.txt'))
    myCentroids,clusterAssing=kMeans(datMat,4)

