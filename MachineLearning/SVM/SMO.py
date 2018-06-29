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

"""随机选择一个alpha"""
def selectJrand(i,m):  #i是alpha的下标，m是所有alpha的数目
    j=i
    while (j==i):
        j=int(random.uniform(0,m))   #random.uniform(x, y)随机生成实数，范围是[x,y)
    return j

"""调整大于H或小于L的alpha值"""
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

"""简化版的SMO"""
def somSimple(dataMatIn,classLabels,C,toler,maxIter):   #5个输入参数：数据集，类别标签，常数C，容错率和最大循环次数
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    b=0
    m,n=shape(dataMatrix)  #m是样本个数，n是特征数
    alphas=mat(zeros((m,1)))  #alpha矩阵，alpha个数等于样本个数
    iter=0  #用于存储在没有任何alpha改变的情况下遍历数据集的次数
    while(iter<maxIter):
        alphaPairsChanged=0   #记录alpha是否已经进行了优化
        for i in range(m):
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b  #multiply是对应元素相乘，*就是矩阵乘法 fXi是预测的f(x)，是预测类别
            Ei=fXi-float(labelMat[i])  #预测类别和真实标签的差值即为误差
            if((labelMat[i]*Ei<-toler)and(alphas[i]<C))or((labelMat[i]*Ei>toler)and(alphas[i]>0)): #如果i样本的预测误差很大且αi不等于0或C就可以进行优化
                j=selectJrand(i,m)  #随机选择一个αj
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b  #计算αj的预测值
                Ej=fXj-float(labelMat[j])  #计算αj的误差
                alphaIold=alphas[i].copy()  #将现在的alpha[i]和alpha[j]相应的保存
                alphaJold=alphas[j].copy()
                #L和H用于将alpha[j]调整到0和C之间
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:  #如果L==H就不做任何调整，直接做下一次的for循环
                    print("L==H")
                    continue
                #计算δ，δ是αj的最优修改量，如果δ>=0就要跳出for循环的当前迭代
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                #用求出的L和H对αj进行调整
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):  #如果αj调整过于轻微，则跳出当前的循环
                    print("j not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])  #αi也做调整，大小同αj方向相反
                #为αi和αj设置一个常数项b
                #由f(x)=∑(m,i=1)αi*yi*xi.T*x+b，可以求得常数项b
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i])and(C>alphas[i]):
                    b=b1
                elif(0<alphas[j])and(C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1  #表示该alpha对进行了优化
                print("iter:{}".format(iter),"i:{}".format(i),",pairs changed{}".format(alphaPairsChanged))
        if(alphaPairsChanged==0):   #遍历数据集结束后，没有一对alpha进行了优化，则要将遍历次数加一
            iter+=1
        else:
            iter=0
        print("iteration number:{}".format(iter))   #否则遍历结束
    return b,alphas  #返回SMO求得的b和α值

if __name__ == '__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=somSimple(dataArr,labelArr,0.6,0.001,40)
    print(b)
    print(alphas[alphas>0])