import jieba
import numpy as np

#读取停用词
def read_from_file(file_name):
    with open(file_name,mode='r') as fp:
        words = fp.read()
    return words


#获取停用词表
def stop_words(stop_word_file):
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        #print(r)
        new_words.append(r)
    return set(new_words)


#获取词袋空间
def get_words_bags(path):
    text=read_from_file(path)
    txt_cut = jieba.cut(text)
    #txt_cut=txt_cut.strip('\n')
    stop_cut = stop_words("/home/opprash/Desktop/datas/data/stop_words.txt")
    #stop_cut = jieba.cut(stop_word_set)
    new_words = []
    for r in txt_cut:
        #if r not in stop_cut:
        new_words.append(r)
    new_words=list(set(new_words))
    new_words.remove('\n')
    return new_words


#onehot编码
def one_hot_encoding(str1,word_bag):
    print(word_bag)
    one_hot=np.zeros(len(word_bag))
    str_cut=jieba.cut(str1)
    for a in str_cut:
        one_hot[word_bag.index(a)]=1
    return one_hot

if __name__ == "__main__":
    bags=get_words_bags('/home/opprash/Desktop/sssss.txt')
    print(type(bags))
    strs='我爱中国'
    s=one_hot_encoding(strs,bags)
    print(s)