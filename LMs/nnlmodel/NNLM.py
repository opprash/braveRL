import glob
import random
import math
import pickle
import numpy as np


# 激活函数
def tanh(o, d):
    x = []
    for i in o:
        x.append(math.tanh(i))
    return x


def get_stopword_list(path):
    """
    载入停用词
    """
    stopword_list = [sw.replace('\n', '')
                     for sw in open(path, 'r', encoding='utf8')]
    return stopword_list


def data_pre(path):
    """
    数据载入，以及完成分词,统计总词数
    """
    import jieba
    content = []
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        # sw_list = get_stopword_list('./data/stop_words.utf8')
        for l in f:
            l = l.strip()
            if (len(l) == 0):
                continue
            l = [x for x in jieba.cut(l) if x not in get_stopword_list(
                './data/stop_words.utf8')]
            content.append(l)
    return content


# 随机生成词向量并分配词id
def creat_wv(wd, m):
    wd = {i: [random.random() for x in range(m)] for i in wd}
    idd = 0
    wid = {}
    for i in wd:
        wid[i] = wid.get(i, 0) + idd
        idd += 1
    # wd['space__']=[random.random() for x in range(m)]
    # wid['space__']=wid.get(i,0)+idd
    return wd, wid


f = glob.glob(r'./data/news/*.txt')
data = []
wd = {}
c = 0
sf = len(f)
for text in f:
    c += 1
    temp = data_pre(text)
    data.extend(temp)
    for t in temp:
        for w in t:
            wd[w] = wd.get(w, 0) + 1
    print(text + ' complete ', end='')
    print(c / sf)
# print(data)
savedata = np.array(data)
swd = np.array(wd)
np.save('./data/sogo_news.npy', savedata)
np.save('./data/myw2vwd.npy', swd)
# data = np.load('./data/sogo_news.npy').tolist()
# 初始化神经网络
h = 100
v = len(wd)
m = 100
n = 4
win = 2
theta = 0.1  # 学习率
# 输入层到隐藏权值，shape=n*m  *  h    n为window的大小，h为隐层神经元个数
H = [[random.random() for j in range(n * m)] for i in range(h)]
H = np.array(H)
d = [random.random() for j in range(h)]  # 隐层偏置 shape=1*h
U = [[random.random() for j in range(h)]
     for i in range(v)]  # 隐层到输出层权值 shape=h*V V为词的总数目
b = [random.random() for j in range(v)]  # 输出层偏置 shape = 1* V
maxtime = 5
sapce = [0 for i in range(m)]  # 空词向量
wvd, wid = creat_wv(wd, m)  # 随机生成词向量和id
sums = len(data)
while (maxtime > 0):
    maxtime -= 1
    # 训练神经网络
    sm = 0
    for s in data:  # s 是一句话
        aa = (sm + 0.0) / sums
        sm += 1
        print('less', end='')
        print(maxtime, end='------------')
        print(aa)
        for w in range(len(s)):  # w是目标词下标
            # 构建输入向量x
            x = []
            inputword = []
            w_id = wid[s[w]]  # 目标词id
            # w_id2 = []#输入词
            for i in range(w - win, w + win + 1):
                # w_id2.append(s[i])
                if i < 0:
                    x.extend(sapce)
                elif i == w:
                    continue
                elif i >= len(s):
                    x.extend(sapce)
                else:
                    x.extend(wvd[s[i]])
                    inputword.append(s[i])

            # ---前向计算------------------------
            # 计算隐层输入
            o = np.dot(x, H.T) + d
            # 计算隐层输出
            a = tanh(o, 1)
            a = np.array(a)
            # 计算输出层输入
            U = np.array(U)
            # H = np.array(H)
            y = np.dot(a, U.T) + b
            y = y.tolist()
            # 计算输出
            p = [math.exp(i) for i in y]
            S = sum(p)
            p = [i / S for i in p]
            # ----前向计算结束------------------------

            # 计算目标函数L
            if p[w_id] != 0:
                L = math.log(p[w_id])
            else:
                L = 2.2250738585072014e-200

            # ----反向传播------------------------
            la = 0
            lx = 0
            ly = [-i for i in p]
            ly[w_id] += 1
            b = np.array(b)
            ly = np.array(ly)
            lb = b + theta * ly
            la = ly[0] * U[0]
            for j in range(1, v):
                la += theta * ly[j] * U[j]
            for j in range(1, v):
                U[j] += theta * la
            lo = [0 for q in range(len(la))]
            lo = np.array(lo)
            for k in range(h):
                lo[k] = (1 - a[k] * a[k]) * la[k]
            lx = np.dot(H.T, lo)
            d += theta * lo
            x = np.matrix(x)
            lo = np.matrix(lo)
            H += theta * np.dot(lo.T, x)
            x += theta * lx
            x = x.tolist()[0]
            for q in range(len(inputword)):
                a = x[0 + i * m:m + i * m]
                for j in range(len(a)):
                    wvd[inputword[q]][j] += a[j]
            # ---反向更新结束
# 保存数据
output = open('./data/myw2v.pkl', 'wb')
pickle.dump(wvd, output)

