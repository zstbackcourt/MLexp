from math import log
import operator


"""测试用数据集"""
def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels




"""计算信息熵"""
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)  #数据集中样本(实例)的总数   len(numpy.array)输出array的第一维中的个数，这里就是样本的个数
    labelCounts={}   #样本标记字典初始化
    #为所有可能的分类创建字典
    for featVec in dataSet:
        currentLabel=featVec[-1]   #字典的键值就是样本矩阵的最后一列数值（即样本标签）
        if currentLabel not in labelCounts.keys(): #如果当前键值不在字典中，则扩展字典并将该键值存入
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  #然后记录当前类别出现的次数
    shannonEnt=0.0  #初始化信息熵
    #Ent=-∑pk*log2pk
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)  #以2为底求对数
    return shannonEnt

"""按照给定特征划分数据集"""
def splitDataSet(dataSet,axis,value):    #三个输入参数：待划分的数据集，划分数据集的特征(即用第几个特征划分)，需要返回的特征的值(即划分条件)
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return  retDataSet

"""选择最好的数据集划分方案"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1   #样本的特征数,dataSet[0]表示第一个样本，因为包括一个标签所以要减一
    baseEntropy=calcShannonEnt(dataSet)   #计算数据集信息熵
    bestInfoGain=0.0  #初始化最优信息增益
    bestFeature=-1   #初始化最好的特征(下标)
    for i in range(numFeatures):
        #创建为一个分类标签列表
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)   #用featList，set()函数创建一个无序不重复元素集
        newEntropy=0.0
        for value in uniqueVals:
            #计算每种划分方法的信息熵
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))   #该划分的信息熵的权值
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy  #计算信息增益
        #选择最好的信息增益
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

"""多数表决(投票法)决定叶子节点的分类，用于已经划分了所有属性，但叶子节点中样本仍然完全属于同一类别"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #排序，按照类别样本数排序
    return sortedClassCount[0][0]

"""递归创建决策树"""
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]    #初始化样本标签列表
    #递归返回条件1：所有样本属于同一类别
    if classList.count(classList[0])==len(classList):  #判断整个数据集的类别是否属于同一类别，若属于同一类别则停止划分
        return classList[0]
    #递归返回条件2，特征全部遍历完，没有将样本分为同一个类别
    if len(dataSet[0])==1:  #数据集中不再有样本特征后停止遍历
        return majorityCnt(classList)  #返回包含样本数最多的类别
    #若特征没有全部遍历完，选择最优特征划分
    bestFeat=chooseBestFeatureToSplit(dataSet)  #当前数据集选取的最优特征（选择bestFeat来划分数据集）这里是一个索引下标
    bestFeatLabel=labels[bestFeat]  #用索引下标对应着找出相应的特征标签
    #开始创建树，myTree保存了树的所有信息
    myTree={bestFeatLabel:{}}  #用字典形式存储返回值
    del(labels[bestFeat])  #将分类过了的分类标签删除
    #创建树，遍历当前选取的特征包含的所有属性值
    featValues=[example[bestFeat] for example in dataSet]
    uniqueValues=set(featValues)  #该特征包含的所有的属性值放入集合，属性值是不重合的
    #在每一个属性值上递归调用createTree
    for value in uniqueValues:
        subLabels=labels[:]  #保存labels数据，防止因为引用方式传递导致原始数据变化
        #递归生成决策树
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)  #得到的返回值插入字典myTree
    return myTree

"""测试"""
if __name__=='__main__':
    myDat,labels = createDataSet()
    print(myDat[0])
    print(len(myDat[0]))







