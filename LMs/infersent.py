import tensorflow as tf
import numpy as np

def bilstm_as_encoder(sent_padded_as_tensor, word_embeddings,
layer_size, hidden_size=100, sent_length=50, embedding_size=300):
    #选取特定的文本特征（输入是一个句子的一些关键词，然后去磁词袋中匹配，有点类似于Onehot编码的过程）
    embed_input = tf.nn.embedding_lookup(word_embeddings,sent_padded_as_tensor)
    #定义前向ltsm隐藏层数量
    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size)
    #定义反向ltsm隐藏层数量
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size)
    #生成biLTSM网络
    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,inputs= embed_input, dtype=tf.float32)
    #拼接rnn输出的第三个维度
    concatenated_rnn_outputs = tf.concat(rnn_outputs, 2)
    #最大池化
    max_pooled = tf.layers.max_pooling1d(concatenated_rnn_outputs, sent_length, strides=1)

    max_pooled_formated = tf.reshape(max_pooled, [-1, 2*hidden_size])

    #初始化参数b和w
    w1 = tf.get_variable(name="w1", dtype=tf.float32, shape=[2*hidden_size, layer_size[0]])
    b1 = tf.get_variable(name="b1", dtype=tf.float32, shape=[layer_size[0]])
    #编码
    encoded = tf.matmul(max_pooled_formated, w1) + b1
    return encoded


def build_graph(inputs1,inputs2,emb_matrix,encoder,embedding_size = 300,layer_size = None,nclasses = 3):


    #将嵌入矩阵转化为张量
    word_embeddings = tf.convert_to_tensor(emb_matrix, np.float32)
    print(word_embeddings)
    # the encoders
    with tf.variable_scope("encoder_vars") as encoder_scope:
        #将输入1编码
        encoded_input1 = encoder(inputs1, word_embeddings, layer_size)
        encoder_scope.reuse_variables()
        #将输入2编码
        encoded_input2 = encoder(inputs2, word_embeddings, layer_size)

    #向量1和向量2做减法
    abs_diffed = tf.abs(tf.subtract(encoded_input1, encoded_input2))
    print(abs_diffed)
    # 向量1和向量2做乘法
    multiplied = tf.multiply(encoded_input1, encoded_input2)
    print(multiplied)
    #将最后得到的几个向量连接起来(混合)
    concatenated = tf.concat([encoded_input1, encoded_input2,abs_diffed, multiplied], 1)
    print(concatenated)
    concatenated_dim = concatenated.shape.as_list()[1]

    #定义全连接层的数量
    fully_connected_layer_size = 512
    with tf.variable_scope("dnn_vars") as encoder_scope:
        wd = tf.get_variable(name="wd", dtype=tf.float32,shape=[concatenated_dim, fully_connected_layer_size])
        bd = tf.get_variable(name="bd", dtype=tf.float32,shape=[fully_connected_layer_size])
    #全连接层的计算
    dnned = tf.matmul(concatenated, wd) + bd
    print(dnned)

    with tf.variable_scope("out") as out:
        w_out = tf.get_variable(name="w_out", dtype=tf.float32,shape=[fully_connected_layer_size, nclasses])
        b_out = tf.get_variable(name="b_out", dtype=tf.float32,shape=[nclasses])
    #定义逻辑输出，3层softmax层
    logits = tf.matmul(dnned, w_out) + b_out

    return logits
