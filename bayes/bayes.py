# coding = gbk

from numpy import *
import re

'''加载词集和分类标签'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']];

    classVec = [0,1,0,1,0,1]; # 1代表侮辱性文字，0代表正常言论

    return postingList,classVec

'''将所有单词去除重复单词后，转换成词汇表'''
def createVocabList(dataSet):
    vocabSet = set([]);
    for document in dataSet:
        vocabSet = vocabSet | set( document )

    return list(vocabSet)

'''将单词转换成词向量，单词存在于词汇表中，则对应位置置为1'''
def words2Vec(vocabList,inputWords):
    returnVec = [0] * len(vocabList)
    for word in inputWords:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print( "the word: %s is not in my Vocabulary!" % word )

    return returnVec

'''训练贝叶斯模型'''
def trainingNB(trainMatrix,classVec):
    numTrainDocs = len(trainMatrix)
    numwords = len(trainMatrix[0])
    fAbusive = sum(classVec)/float(numTrainDocs) #类别i的文档数除以总文档数
    p0Num = ones(numwords); p1Num = ones(numwords);#利用以2为底的log来求解（避免连乘，精度损失问题），利用log求解时矩阵初始化为1
    p0Denom = 2.0; p1Denom = 2.0#基数初始化为2

    for i in range(numTrainDocs):
        if classVec[i] == 1:
            p1Num += trainMatrix[i];#记录该类别中出现的所有单词 侮辱词汇
            p1Denom += sum(trainMatrix[i]);#统计该类别单词总数
        else:
            p0Num += trainMatrix[i];
            p0Denom += sum(trainMatrix[i])

    p0Vec = log (p0Num/p0Denom); #条件概率
    p1Vec = log (p1Num/p1Denom);

    return p0Vec,p1Vec,fAbusive

'''利用贝叶斯进行分类'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1) #求待测词向量是侮辱词集的概率
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1) #求待测词向量不是侮辱词集的概率

    if(p1 > p0 ): return 1;
    else: return 0;

def testingNB():
    listPosts,classLabels = loadDataSet();
    vocablist = createVocabList(listPosts);
    traMat = [];

    '''将词汇表转换成词条矩阵'''
    for document in listPosts:
        traMat.append( words2Vec(vocablist,document) )

    p0Vec,p1Vec,pAb = trainingNB(traMat,classLabels)#利用词条矩阵和类别求贝叶斯概率

    testEnty=['dog'];
    docVec = words2Vec(vocablist,testEnty ); #将待测文字转换成词条向量
    classes = classifyNB(docVec,p0Vec,p1Vec,pAb);
    return classes

'''字符串解析函数'''
def testParse(bigString):
    listOfTokens = re.split(r'\\W*',bigString);
    return [tok.lower() for tok in listOfTokens if len(tok) > 2 ]

def spamTest():
    docList = []; classes = []; fullTextList = [];

    '''生成输入词集和类别标签'''
    for i in range(1,26):
        fr = open('E:\machinelearninginaction\Ch04\email\spam\%d.txt'% i );
        wordList = testParse( fr.read() );
        docList.append(wordList)
        classes.append(1)
        fr = open('E:\machinelearninginaction\Ch04\email\ham\%d.txt' % i );
        wordList = testParse(fr.read());
        docList.append(wordList)
        classes.append(0)

    vocabList = createVocabList(docList) #将词集合转换成词汇表
    trainSet = list( range(50) )
    testSet = [];

    '''随机生成测试数据集合，并从训练结合中删除测试数据'''
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex] );
        del( trainSet[randIndex])

    trainMat = []; trainClasses = [];

    '''将训练数据集转换成词向量，并获得对应的标签'''
    for docIndex in trainSet:
         trainMat.append( words2Vec(vocabList,docList[docIndex]) )
         trainClasses.append(classes[docIndex])

    p0v,p1v,pSpam = trainingNB(trainMat,trainClasses) #利用数据训练贝叶斯得到
    errorCount = 0.0

    '''利用训练得到的数据,进行分类'''
    for docIndex in testSet:
        wordVec =  words2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVec,p0v,p1v,pSpam) != classes[docIndex]:
            errorCount += 1;
            print(classes[docIndex])
            print(docList[docIndex])

    print(float(errorCount)/len(testSet))