import math


def dis(a, b):
    s = 0
    for i in range(len(a)):
        t = a[i] - b[i]
        t = t * t
        s += t
    return math.sqrt(s)


import pickle

inputt = open('./data/myw2v.pkl', 'rb')
wd = pickle.load(inputt)
a = wd['记者']
b = wd['公司']
c = wd['企业']
d = wd['交易']
e = wd['支付']
print(dis(a, b))
print(dis(b, c))
print(dis(e, d))
print(dis(a, e))