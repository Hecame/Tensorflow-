#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):

    #将其内定义的节点设在计算图tg中
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        #返回y中1维列表中最大值所在索引值
        preValue = tf.argmax(y, 1)

        #实例化带滑动平均值的saver对象
        variables_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variables_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            #加载模型
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1



def pre_pic(picName):
    #输入图像预处理
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)
    #调整图像尺寸为28*28像素,Image.ANTIALIAS参数表用消除锯齿的方法调整图像尺寸
    im_arr = np.array(reIm.convert('L'))    #将图像转化为灰度图,再转化为矩阵
    threshold = 58          #二值化处理阀值
    #将白底黑字转化为黑底白字形式,再二值化处理图像,以滤掉噪声
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])   #将矩阵维度转为[1,784]
    nm_arr = nm_arr.astype(np.float32)  #将矩阵元素转为np.float32格式
    img_ready = np.multiply(nm_arr, 1.0/255.0) #将矩阵内元素转为0~1间的浮点数

    return img_ready



def application():
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        testPic = raw_input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print "The prediction number is:", preValue

def main():
    application()

if __name__ == '__main__':
    main()
