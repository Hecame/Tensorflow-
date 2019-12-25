#coding:utf-8

import tensorflow as tf

IMAGE_SIZE = 28     #图片尺寸
NUM_CHANNELS =1     #图片通道数
CONV1_SIZE = 5      #卷积核1尺寸
CONV1_KERNEL_NUM = 32   #卷积核1的个数
CONV2_SIZE = 5      #卷积核2尺寸
CONV2_KERNEL_NUM = 64   #卷积核2的个数
FC_SIZE = 512       #全连接层1节点数
OUTPUT_NODE = 10    # mnist输出的标签矩阵为[1,10]

def get_weight(shape, regularizer):
    #去掉过大偏离点的正态分布
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: 
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):
    #卷积操作,x为输入矩阵,w为卷积核,行列步长为1
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #最大池化操作,核尺寸为2x2,行列步长为2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1,conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    #将pool2从3维张量转为2维张量
    pool_shape = pool2.get_shape().as_list()    #获取pool2的维度以列表形式赋值给pool_shape
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]   #节点数=行数*列数*通道数
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])    #pool_shape[0]是batch_size

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    #如果是训练模型过程,舍弃50%全连接节点的参数训练,以加快训练进度
    if train: fc1 = tf.nn.dropout(fc1, 0.5) 

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b

    return y
