from numpy import *
import random

"""测试用数据加载"""
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

"""建立一个类来保存所有数据的值"""
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros(self.m,1))
        self.b=0
        self.eCache=mat(zeros(self.m,2))  #用来存储误差E，eCache的第一列给出eCache是否有效地标志位，第二列给出的实际的E值

"""辅助函数用来计算误差E"""
def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*os.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

"""用于选择第二个alpha(内循环的alpha)"""
"""目标是选择合适的第二个alpha保证在每次优化中采用最大步长"""
def selectJ(i,oS,Ei):
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]   #设置Ei有效（1）
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]   #构造一个非零列表".A"将矩阵转化为列表，非零E值对应的alpha值
    if(len(validEcacheList))>1:
        for k in validEcacheList:   #validEcacheList的所有值上循环找到使得改变最大的那个值
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:  #如果是第一次循环就随机选择一个alpha
        j=selectJ(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

"""误差值并存入缓存，在对alpha优化时会用到"""
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

"""调整大于H或小于L的alpha值"""
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

"""完整SMO中的优化例程"""
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.alphas[i]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L==H:
            print("L++H")
            return 0
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if eta>=0:
            print("eta>=0")
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not mving enough")
            return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaJold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaJold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i])and(oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j])and(oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

"""完整的SMO外循环代码"""
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print("fullSet,iter:{}".format(iter),"i:{}".format(i),"pairs changed:{}".format(alphaPairsChanged))
                iter+=1
        else:
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("non-bound,iter:{}".format(iter),"i:{}".format(i),"pairs changed:{}".format(alphaPairsChanged))
                iter+=1
        if entireSet:
            entireSet=False
        elif(alphaPairsChanged==0):
            entireSet=True
        print("iteration number:{}".format(iter))
    return oS.b,oS.alphas


"""求w"""
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w