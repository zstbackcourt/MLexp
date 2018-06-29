from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


"""测试用训练数据集"""
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

"""KNN分类算法"""
def classify0(inX,dataSet,labels,k):     #分类的输入向量inX，训练样本集dataSet，标签向量labels，k近邻数目
    dataSetSize=dataSet.shape[0]  #获取训练集中的样本个数,因为shape返回的是一个元组
    """要预测的样本和训练集中各样本的距离"""
    diffMat=tile(inX,(dataSetSize,1)) - dataSet  # 将向量inX在行方向上复制dataSetSize次，形成一个矩阵，然后和dataSet做矩阵减法，所得的结果就是各坐标的差值
    sqDiffMat=diffMat**2   #矩阵元素求平方
    sqDistances=sqDiffMat.sum(axis=1) #sum(axis=1)将矩阵每一行元素相加，压缩成一行
    distances=sqDistances**0.5 #矩阵每个元素开平方，得到预测样本和各训练样本之间的欧式距离
    sortedDistIndicies=distances.argsort()  #返回各距离的排序（从小到大）
    classCount={}
    """对预测样本进行分类"""
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]  #选取距离最小的k点
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #放入字典classCount，字典的key即为该距离对应的样本的标签，该分类数加一
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #对字典各元素从大到小排列，并以元组形式返回
    return sortedClassCount[0][0]  #排在第一位的key/value对应的分类，即为发生频率最高的标签

"""归一化特征的函数"""
def autoNorm(dataSet):
    minVals=dataSet.min(0) #每列的最小值
    maxVals=dataSet.max(0)  #每列的最大值
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))  #初始化归一矩阵(全零)
    m=dataSet.shape[0]  #数据集中的样本个数
    """归一化处理"""
    normDataSet=dataSet-tile(minVals,(m,1))  #当前值减最小值
    normDataSet=normDataSet/tile(ranges,(m,1))   #除以取值范围
    return normDataSet,range,minVals


"""将Hellen的约会数据转化成Numpy的解析程序"""
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines() #读文件
    numberOfLines=len(arrayOLines) #获得文件的行数
    returnMat=zeros((numberOfLines,3))  #创建一个全0矩阵大小为numberofLines*3
    classLabelVector=[]  #创建分类标签向量
    index=0
    for line in arrayOLines:
        line=line.strip() #移除字符串头尾指定的字符（默认为空格或换行符）
        listFromLine=line.split('\t')  #通过指定分隔符对字符串进行分割，返回分割后的字符串列表
        returnMat[index,:]=listFromLine[0:3]   #将listFromLine[0:3]切片的元素对应放到returnMat第index行中
        classLabelVector.append(int(listFromLine[-1]))  #把listFromLine的最后一个元素（标签）放进标签向量
        index+=1
    return returnMat,classLabelVector

"""分类器针对约会网站的测试代码"""
def datingClassTest():
    hoRatio=0.1   #定义测试样本占整个样本集的比例为10%
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)  #测试集样本数
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("预测的分类为:{}".format(classifierResult),"\t真实的分类为:{}".format(datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print("整体的错误率为：{}".format(errorCount/float(numTestVecs)))



"""测试"""
if __name__=='__main__':
    # group,labels=createDataSet()
    # result = classify0([0,0],group,labels,3)
    # print(result)
    # datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    # print(datingDataMat)
    # print(datingLabels)
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.show()
    datingClassTest()