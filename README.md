# My first share project of nlp

## that's cool!
## 这是最新版本，之前的库更名为braveRL-past，可以去https://github.com/opprash/braveRL查看，之前那个库不在误会，以后的代码基本都会提交在这个库里面。

## onehot编码
onehot本质上属于词典模型，最终一句话的向量维数是词袋中词的总数，假设有几句话：

1. 我爱中国
2. 爸爸妈妈爱我
3. 爸爸妈妈爱中国
首先，将语料库中的每句话分成单词，并编号：

1：我      2：爱      3：爸爸      4：妈妈      5：中国

然后，用one-hot对每句话提取特征向量：

![onehot](https://github.com/opprash/braveRL/blob/master/datas/one_hot.png)

所以最终得到的每句话的特征向量就是：

1. 我爱中国 -> 1，1，0，0，1
2. 爸爸妈妈爱我 -> 1，1，1，1，0
3. 爸爸妈妈爱中国 -> 0，1，1，1，1

## tfidf提取特征
TF-IDF（term frequency–inverse document frequency，词频-逆向文件频率）是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用加权技术。
1. TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
2. TF-IDF的主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
* TF是词频(Term Frequency)     
词频（TF）表示词条（关键字）在文本中出现的频率。  
这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件。  
![tf](https://github.com/opprash/braveRL/blob/master/datas/tf.png)  

* IDF是逆向文件频率(Inverse Document Frequency)  
逆向文件频率 (IDF) ：某一特定词语的IDF，可以由总文件数目除以包含该词语的文件的数目，再将得到的商取对数得到。如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。  
![idf](https://github.com/opprash/braveRL/blob/master/datas/idf.png)
其中，|D| 是语料库中的文件总数。 |{j:ti∈dj}| 表示包含词语 ti 的文件数目（即 ni,j≠0 的文件数目）。如果该词语不在语料库中，就会导致分母为零，因此一般情况下使用 1+|{j:ti∈dj}|   
* TF-IDF实际上是：TF * IDF  
某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。  
![tfidf](https://github.com/opprash/braveRL/blob/master/datas/tfidf.png)
3. TF-IDF应用  
（1）搜索引擎  
（2）关键词提取  
（3）文本相似性  
（4）文本摘要  
## word2vec
要理解word2vec原理先来看张图：    
![word2vec](https://github.com/opprash/braveRL/blob/master/datas/word2vec.png)   
以上就是word2vec的结构，首先将词向量进行一次嵌入然后根据这个嵌入向量去获得目标。   
假如有一句话：今天 下午 我 不 上班
1. 根据上下文预测目标单词：
* 要预测“我”这个目标单词，可以通过上文的”今天“，”下午“来预测同时也可以使用下文的“不” ，”上班“两个单词来预测，一般将上下文一起结合起来预测比较好因为这样结合了上下文的语境。
这种模型称为CBOW，模型包含三层：输入层,映射层和输出层.CBOW模型中的w(t)为目标词,在已知它的上下文w(t-2),w(t-1),w(t+1),w(t+2)的前提下预测词w(t)出现的概率,即：p(w/context(w))目标函数为：  
![CBOW](https://github.com/opprash/braveRL/blob/master/datas/CBOW.png)
2. 根据当前词来预测上下文
* 可以通过“我”来预测“今天”，“下午”,"不"，“上班”这几个字，这种模型称为skim_gram，skim-gram模型同样包含三层：输入层,映射层和输出层.Skip-Gram模型中的w(t)为输入词,在已知词w(t)的前提下预测词w(t)的上下文w(t-2),w(t-1),w(t+1),w(t+2),条件概率写成：p(context(w)|w)  
![skip-gram](https://github.com/opprash/braveRL/blob/master/datas/skip_gram.png)  

## infersent
整体的架构如图：  
![infersent](https://github.com/opprash/braveRL/blob/master/datas/infersent.png)  

这个框架最底层是一个Encoder，也就是最终要获取的句子向量提取器，然后将得到的句子向量通过一些向量操作
后得到句子对的混合语义特征，最后接上全连接层并做SNLI上的三分类任务，做过句子匹配任务的一定知道，这个
框架是一个最基本（甚至也是最简陋）的句子匹配框架。对于底层的Encoder来说，论文作者分别尝试了7种模型，
然后分别以这些模型作为底层的Encoder结构，然后在SNLI上进行监督训练。训练完成后，在新的分类任务上进
行评估，最后发现当Encoder使用BiLSTM with max pooling结构时，对于句子的表征性能最好。  

