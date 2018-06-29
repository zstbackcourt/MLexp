from numpy import *




def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))    #将每一行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

"""切分数据集（注意这里书上错了）"""
def binSplitDataSet(dataSet,feature,value):#参数：数据集、待切分的特征、该特征的某个值
    #通过数组过滤方式将数据集切分得到两个子集返回
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]    #选出指定特征feature满足大于特征值value的样本数据
    mat1=dataSet[nonzero(dataSet[:, feature]<=value)[0],:]  #选出指定特征feature满足小于等于特征值value的样本数据
    return mat0,mat1



"""负责生成叶结点"""
#当chooseBestSplit函数确定不再对数据进行切分时，将调用该函数来得到叶结点的模型。在回归树中，该模型其实就是目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

"""负责计算目标变量的平方误差"""
def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]    #均方差函数var()，因为要返回总方差，故要乘以样本个数

"""树构建函数"""
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #leafType是对创建叶结点的引用；errType是对总误差方差计算函数的引用；ops是用户定义的参数构成的元组，用于树构建
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)     #根据基尼指数选择最好的特征用于划分
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)   #递归构建左子树
    retTree['right']=createTree(lSet,leafType,errType,ops)  #递归构建右子树
    return retTree

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #tolS和tolN用于控制函数的停止时机
    tolS=ops[0]   #容许的误差下降值
    tolN=ops[1]   #切分的最少样本数
    #将预测值y(特征值）/分类类别转化成一个列表(dataSet[:,-1].T.tolist()[0])
    #set函数将这个列表转化成集合，即特征值不同的才会被放入集合
    #len计算集合长度，如果为1说明不同剩余特征值的数目为1，那么就不需要在切分，只要直接返回
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)    #用leafType对数据集生成叶结点
    m,n=shape(dataSet)   #n是特征数和y的和
    S=errType(dataSet)
    bestS=inf
    bestIndex=0
    bestValue=0
    #在所有可能的特征及其可能的取值上遍历
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)  #切分数据集
            if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):   #判断是否还需继续切分
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if (S-bestS) < tolS:
        return None,leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):   #如果提前终止条件均不满足则返回切分特征和特征值
        return None,leafType(dataSet)
    return bestIndex,bestValue

if __name__=='__main__':
    myDat=loadDataSet('ex00.txt')
    myMat=mat(myDat)
    regTree=createTree(myMat)
    print(regTree)


