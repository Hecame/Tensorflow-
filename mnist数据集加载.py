#coding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)
#以读热码的形式存取路径 ./data/ 下的文件,如不存在将自动下载文件保存到该路径 


#返回各子集样本数
#print "train data size:", mnist.train.num_examples #训练集
#print "validation data size:", mnist.validation.num_examples #验证集
#print "test data size:", mnist.test.num_examples   #测试集

#返回标签和数据
#mnist.train.labels[0]  #表训练集第0张图的标签矩阵
#mnist.train.images[0]  #表训练集第0张图的图像矩阵

BATCH_SIZE = 200    #定义每次输入数据集数量
# 在所有训练数据集中随机取出 BATCH_SIZE 数量的数据,
# 其中xs是图像矩阵[1,28*28=784],ys是标签矩阵[1,10]
xs, ys = mnist.train.ext_batch(BATCH_SIZE)
print "xs shape:", xs.shape
print "ys shape:", ys.shape
