from collections import defaultdict
import math
import operator
import re
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""
函数说明:创建数据样本
Returns:
    dataset - 实验样本切分的词条
    classVec - 类别标签向量
"""
def loadDataSet():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表好，0代表不好
    return dataset, classVec


"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""
def feature_select(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储没个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数

    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1

    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}

    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select

def deal_datas(path,length):
    strs=[]
    str1=[]
    label=[]
    #s = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    s1 = {'体育': 1, '财经': 2, '房产': 3, '家居': 4, '教育': 5, '科技': 6, '时尚': 7, '时政': 8, '游戏': 9, '娱乐': 0}
    path1 = path
    with open(path1, encoding='utf-8', mode='r') as file:
        line = file.readline()
        while line:
            strs.append(line)
            line = file.readline()
        file.close()
    for each in range(length):
        for s in range(10):
            num=each+s*length
            fin=strs[num]
            left = re.split('\t', fin)
            print(left)
            label.append([s1[left[0]]])
            str1.append(left[1])
    return label,str1


"""
函数说明：使用sklearn实现的TF-IDF算法
Parameters:
     path:文本路径
     length:文本长度
     max_features:最大特征数
Returns:
     x_train_weight:文本tf-idf
     np_label：标签
"""
def stop():
    stpwrdpath = "/home/opprash/Desktop/project/lda/datas/stop_words.txt"
    stpwrd_dic = open(stpwrdpath, 'r')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    return stpwrdlst


def skl_tfidf(path,length,max_features):
    enc = preprocessing.OneHotEncoder()
    enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    labels, docs = deal_datas(path, length)
    labels = np.array(labels)
    np_label = enc.transform(labels).toarray()
    vectorizer = CountVectorizer(stop_words=stop(), max_features=max_features)
    # 该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(docs))
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    x_train_weight = tf_idf.toarray()
    return x_train_weight, np_label


if __name__ == '__main__':
    data_list, label_list = loadDataSet()  # 加载数据
    features = feature_select(data_list)  # 所有词的TF-IDF值
    print(features)
    print(len(features))
