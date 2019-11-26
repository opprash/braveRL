# Hierarchically-Refined Label Attention Network for Sequence Labeling
论文连接：arxiv.org/abs/1908.08676
源码连接： Nealcly/BiLSTM-LAN（https://github.com/Nealcly/BiLSTM-LAN）
## 背景
CRF在词性标注，命名实体识别任务中取得了比较好的结果，但是随着深度学习的应用，多数情况下BiLTSM-CRF并没有比不对输出序列剑魔的
BiLSTM-softmax取得更好的结果。其中一个可能的原因是神经网络已经有很想的序列信息编码能力，在这样的情况下，引入CRF可能对效果的提升并不是那么理想。  
为了更好的对输出序列编码，改论文中作者提出了一种逐层改进的给予标签注意力的网络(HR-lan)。该模型通过利用标签知识，更好的捕捉标签之间长期依赖的关系。
在因为词性标注命名实体识别和组合范畴语法超标注的任务上取得不错的效果。
如下图所示，该图是词性标注的示例，对输入句子“ they can fish and also tomatos here”的标注。
在第一层中模型通过局部信息对每个单词进行判断，这里它更倾向于将“can”判断为情态动词(MD)，“fish”被判断为
动词(VB),在经过多层LAN信息交互以后，长期标签依赖关系被考虑以后，"tomatoes"为名词的信息帮助模型对 "can" 和 "fish" 的词性进行重新判断，
认定 "can" 和 "fish" 应为动词(VB)和名词(NN)。    
![lan1](https://github.com/opprash/braveRL/blob/master/datas/lan/lan1.png)   
## model(模型)
下图中包含了两层的BiLTSM-LAN。每一个BiLTSM-LAN对序列信息进行编码的BiLTSM Encoding Layer和对标签信息进行推理的Label Attention Inference Layer
组成。  
![lan2](https://github.com/opprash/braveRL/blob/master/datas/lan/lan2.png)   
BiLSTM Encoding Layer 为标准的 BiLSTM，定义其输出为$H^w=h_1^w,h_2^w,...,h_n^w$。
Label Attention Inference Layer首先通过计算词表示$H^w∈R ^{n*d_h}$与所有候选标签$x^l=x_1^l,x_2^l,...,x_{|L|}^l∈R^{|L|*d_n}$的attention生成，$α∈R ^{n*|L|}$,其中$n$为序列长度,${|L|}$为标签个数，$d_h$为BiLTSM的隐层纬度，$α$表示没歌词对每个信息的概率分布。最终将蕴含标签信息的$H^l=αx^l$与蕴含词信息的$H^w$拼接并输入到下一个BiLSTM-LAN层没在之后的BiLSTM-LAN中输入文本序列的表示和输出标签的序列表示分布同时被BiLSTM编码，底层的BiLSTM-LAN学习到局部的信息，顶层的BiLSTM-LAN学习到更加全局的信息。在最后一层，BiLSTM-LAN直接根据输出预测每个词的标签。  
BiLSTM-LAN可以看出是BiLSTM-softmax的变种，一层BiLSTM-LAN与一层BiLSTM-softmax完全相同，然而多层BiLSTM-softmax仅仅叠加BiLSTM以前更好的编码输入序列，BiLSTM-LAN可以理解为既叠加了BiLSTM也叠加了softmax，用来更好的学习输入和输出序列的表示。
## 实验
本文在词性标注(WSJ, UD v2.2)，命名实体识别(OntoNotes 5.0)和组合范畴语法超标注(CCGBank)上进行了实验。  
![lan3](https://github.com/opprash/braveRL/blob/master/datas/lan/lan3.png)   

![lan4](https://github.com/opprash/braveRL/blob/master/datas/lan/lan4.png)  

![lan5](https://github.com/opprash/braveRL/blob/master/datas/lan/lan5.png)  

![lan6](https://github.com/opprash/braveRL/blob/master/datas/lan/lan6.png)  

![lan7](https://github.com/opprash/braveRL/blob/master/datas/lan/lan7.png)  

其中，*表示利用多任务与半监督学习取得的结果  
## 分析
### 标签可视化
论文使用t-SNE对词性标注的标签向量进行了可视化分析。  
![lan4](https://github.com/opprash/braveRL/blob/master/datas/lan/lan8.png)   
练开始前，所有标签随机分散到空间内。模型训练5轮后，可以看到"NNP"和"NNPS"，"VBD"和"VBN"等相似词性聚集到一起。在训练38轮后，几乎所有相似的词性被聚集到了一起，例如"VB","VBD","VBN","VBG"和"VBP"。  
### 超标签复杂度
为了验证BiLSTM-LAN捕捉长距离标签依赖关系的能力，论文中对不同复杂度的超标签标注准确率进行了分析。越复杂的超标签需要更长期的标签依赖关系进行判断。随着复杂度的增加，BiLSTM-CRF 并没有比 BiLSTM-softmax
 表现的好，然而 BiLSTM-LAN 表现显著高于其他模型。  
![lan9](https://github.com/opprash/braveRL/blob/master/datas/lan/lan9.png)   
## 总结
理论和序列标注实验证明，BiLSTM-LAN通过对所有候选标签进行编码的方式，很好的捕捉了标签间的长期依赖关系，并在一定程度上面解决了标注偏执问题。