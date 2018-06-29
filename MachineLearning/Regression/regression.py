from numpy import*

"""数据导入函数"""
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1   #特征数包括了x0
    dataMat=[]      #数据矩阵（其实是包括了x0的X矩阵）
    labelMat=[]     #标签矩阵
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')   #每一行是一个样本的数据x0,x1,y.注意x0恒为1，实际上是人为加在b前面的，用于方便矩阵运算
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)              #每一个样本的x0~xn加入矩阵X
        labelMat.append(float(curLine[-1]))  #每个样本的标签加入向量Y
    return dataMat,labelMat   #返回X，Y


"""标准线性回归函数"""
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat   #计算 X^T*X
    if (linalg.det(xTx)==0.0):    #linalg模块包含线性代数的函数 det用于计算矩阵的行列式
        print("该矩阵是奇异的，没有逆矩阵")
        return
    ws=xTx.I*(xMat.T*yMat)   #计算w=(X^T*X)^-1*X^Ty
    return ws


def lwlr(testPoint,xArr,yArr,k=1.0):   #testPoint为待预测点
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))    #创建对角阵
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))   #随着样本点与待预测点距离的递增，权重将以指数级衰减，k用于控制衰减速度
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print("该矩阵是奇异矩阵，没有逆矩阵")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))    #计算系数
    return testPoint*ws                #返回预测值


"""岭回归"""
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam    #X^T*X+λE
    if linalg.det(denom)==0.0:
        print("该矩阵是奇异矩阵，没有逆矩阵")
        return
    ws=denom.I*(xMat.T*yMat)
    return ws                     #返回w

"""计算平方误差"""
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

"""矩阵正规化"""
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)  #求解均值
    inVar = var(inMat,0)    #var()求解方差 0压缩行，对各列求方差，返回 1* n 矩阵；1缩列，对各行求方差，返回 m *1 矩阵
    inMat = (inMat - inMeans)/inVar   #矩阵标准化
    return inMat

"""前向逐步线性回归"""
def stageWise(xArr,yArr,eps=0.01,numIt=100):    #eps表示每次迭代需要调整的步长，numIt表示迭代次数
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)    #mean()求取均值：0压缩行，对各列求均值，返回 1* n 矩阵；1缩列，对各行求均值，返回 m *1 矩阵
    yMat=yMat-yMean       #0均值，即每个数据减去均值
    xMat=regularize(xMat)  #矩阵标准化
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError=inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
        wsMax=wsTest
        ws=wsMax.copy()
    returnMat[i,:]=ws.T




if __name__=='__main__':

