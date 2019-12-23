#coding:utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import mnist_generateds

TEST_INTERVAL_SECS = 5      #每轮循环测试时间为5秒
TEST_NUM = 10000            #!测试数据集总样本数

def test():
    
    with tf.Graph().as_default() as g:
        #其内定义的节点在计算图g中
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        #
        ema_restore = ema.variables_to_restore()
        #实例化可还原滑动平均值的saver对象,此后在会话中恢复模型后,所有参数将使用其滑动平均值
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #获取单位 batch 组的测试数据
        img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, isTrain=False)


        while True:
            with tf.Session() as sess:
                #尝试加载ckpt模型
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 恢复模型到当前会话
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 获取原训练轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    #开启线程协调器
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    xs, ys = sess.run([img_batch, label_batch])
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    #accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training steps, test accuracy = %g" % (global_step, accuracy_score))

                    #关闭线程协调器
                    coord.request_stop()
                    coord.join(threads)

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    #mnist = input_data.read_data_sets("./data/", one_hot=True)
    #test(mnist)
    test()

if __name__ == '__main__':
    main()
