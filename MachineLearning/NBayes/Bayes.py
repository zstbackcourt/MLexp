from numpy import *

"""词表到向量的转换函数"""
def loadDataSet():
    postingList=[
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    classVec=[0,1,0,1,0,1]  #上述六句话（词表）类型向量，1代表侮辱性言论，0代表正常言论
    return postingList,classVec

"""创建一个包含在所有文档中出现的不重复词的列表"""
def createVocabList(dataSet):
    vocabSet=set([])  #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet | set(document) #创建并集
    return list(vocabSet)

"""判断词汇表中的单词是否在输入文档中出现"""
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)  #创建一个长度和词汇表一样的全0向量
    #循环遍历，查看文档inputSet中的单词是否出现在词汇表中
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1  #将输出向量中相应的位置置为1
        else:
            print("{}在词汇表中没有出现".format(word))
    return returnVec  #返回向量

"""朴素贝叶斯训练器函数"""
def trainNB0(trainMatrix,trainCategory):  #输入参数为文档矩阵和每篇文档类别标签所构成的向量
    #注意这里的文档矩阵实际上是词汇表中的单词在每一个文档中是否出现的判断矩阵，是由setOfWords2Vec函数求得的
    numTrainDocs=len(trainMatrix)  #训练的文档数
    numWords=len(trainMatrix[0])   #词汇表中包含的所有单词数
    pAbusive=sum(trainCategory)/float(numTrainDocs)  #计算侮辱性文档的概率，因为1为侮辱，0为正常，所以sum侮辱性的数目
    #初始化概率
    p0Num=zeros(numWords)   #由所有单词数来初始化用于计数正常单词数的向量
    p1Num=zeros(numWords)   #由所有单词数来初始化用于计数侮辱性单词数的向量
    p0Denom=0.0             #正常文档的总词数
    p1Denom=0.0             #侮辱性文档的总词数
    for i in range(numTrainDocs):  #循环遍历文档
        if trainCategory[i]==1:    #如果该文档标签被定义为侮辱性文档
            p1Num+=trainMatrix[i]   #就将trainMatrix中的向量值加到向量p1Num中，因为侮辱性词语值为1，正常词语值为0
            p1Denom+=sum(trainMatrix[i])   #增加侮辱性文档的总词汇数
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Denom     #侮辱性词汇的概率
    p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive


"""构建朴素贝叶斯分类器函数"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


"""测试"""
if __name__=='__main__':
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    print(myVocabList)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print(trainMat)
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    print(pAb)
    print(p0V)
    print(p1V)