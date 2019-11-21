#%matplotlib inline
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data


words = read_data("./text8.zip")
print("Data size %d" % len(words))

vocabulary_size = 50000

#将词转化为数字
def build_dataset(words):
    #提取最常见的单词词汇
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 为每一个单词创建一个字典key为词汇中所有单词，value是这个单词出现的频率0代表位置，1，代表第一频繁2代表第二频繁一次类推
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    #遍历每一个单词，然后根据上一步的词汇表将词汇表现成频率的形式
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)

    count[0][1] = unk_count
    #压缩字典，使我们能够根据单词的唯一整数标识符查找单词
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)

print("Most common words (+UNK)", count[:5])

print("Sample data", data[:10])

del words  # Hint to reduce memory


data_index = 0

#定义n-gram和批度
def generate_batch(batch_size, num_skips, skip_window):
      global data_index
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      #设置缓冲区的大小，一般是上下问窗口大小加1（这个1为单词本身）
      span = 2 * skip_window + 1 # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)
      for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
      for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                  while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                  targets_to_avoid.append(target)
                  batch[i * num_skips + j] = buffer[skip_window]
                  labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
      return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# 我们选择一个随机验证集来对最近的邻居进行采样
# 在这里，我们将验证样本限制为具有较低数字ID的词

valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # 定义数据集
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    #创建输入层和隐藏层的权重矩阵
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    #定义softmax层的连接矩阵（隐藏层和输出层的参数矩阵）
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    #定义参数b
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # 对数据集进行onehot处理
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # 计算损失，使用消极的标签
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # ＃注意：优化器将优化softmax_weights和嵌入。
    # ＃这是因为嵌入被定义为一个可变的数量和。
    # ＃优化器的`minim`方法默认会修改所有变量的数量。
    # ＃这有助于张量传递。
    #使用最低损失能优化参数，（反向传播修正权重矩阵）
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # 计算minibatch示例和所有嵌入之间的相似度。
    # 我们使用余弦距离：

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


#运输tensorflow流程每1000次计算一下平均损失
num_steps = 50000
with tf.Session(graph=graph) as session:
      #初始化变量
      tf.global_variables_initializer().run()
      print('Initialized')
      average_loss = 0
      #计算每一步迭代的损失
      for step in range(num_steps):
            batch_data, batch_labels = generate_batch(
              batch_size, num_skips, skip_window)
            feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                  if step > 0:
                        average_loss = average_loss / 2000
                  # The average loss is an estimate of the loss over the last 2000 batches.
                  print('Average loss at step %d: %f' % (step, average_loss))
                  average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                  sim = similarity.eval()
                  for i in range(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                              close_word = reverse_dictionary[nearest[k]]
                              log = '%s %s,' % (log, close_word)
                        print(log)
      final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
    pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)