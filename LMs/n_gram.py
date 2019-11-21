from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import jieba

# 停用词，可能会用到
def stop_words(path):
    #stpwrdpath = path
    stpwrd_dic = open(path, 'r')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    return stpwrdlst


"""
函数说明：sk-learn实现n-gram算法
Parameters:
     path:需要处理的文本路径
     n:n-gram中的n即元数
Returns:
     restult:n-gram的结果
"""

def n_grams(path, n):
    data=[]
    with open(path) as f:
        texts=f.readlines()
    for each in texts:
        s=" ".join(jieba.lcut(each))
        data.append(s)

    vec = CountVectorizer(min_df=1, ngram_range=(n, n))
    metric = vec.fit_transform(data)
    vec.get_feature_names()
    restult = metric.toarray()
    return restult


def test():
    data = ["他用报话机向上级呼喊：“为了祖国，为了胜利，向我开炮！向我开炮！",
            "记者：你怎么会说出那番话？",
            "韦昌进：我只是觉得，对准我自己打，才有可能把上了我哨位的这些敌人打死，或者打下去。"]
    data = [" ".join(jieba.lcut(e)) for e in data] # 分词，并用" "连接
    vec = CountVectorizer(min_df=1, ngram_range=(2,2))
    metric = vec.fit_transform(data)
    vec.get_feature_names()
    s = metric.toarray()
    print(s)
    df = pd.DataFrame(metric.toarray(), columns=vec.get_feature_names()) # to DataFrame
    df.head()
    print(df.head())

if __name__ == '__main__':
    path='/home/opprash/Desktop/text.txt'
    print(n_grams(path,2))
    test()