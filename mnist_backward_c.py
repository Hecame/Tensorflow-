#coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os


BATCH_SIZE = 200    #定义每次输入数据集数量
LEARNING_RATE_BASE = 0.1        #学习率初始值
LEARNING_RATE_DECAY = 0.99      #学习率衰减率
REGULARIZER = 0.0001            #正则化权重
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99     #滑动平均衰减比较值
MODEL_SAVE_PATH = "./model/c"    #模型保存路径
MODEL_NAME = "mnist_model_c"      #模型保存名称

def backward(mnist):

    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, REGULARIZER)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            CKPT_NODE = 1

            global_step_c = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            global_step = tf.Variable(global_step_c, trainable=False)
            steps = STEPS - global_step_c
            print("steps = %d, global_step_c = %s " % (steps, global_step_c))

            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            #ema_restore = ema.variables_to_restore()
            #实例化可还原滑动平均值的saver对象,此后在会话中恢复模型后,所有参数将使用其滑动平均值
            #saver = tf.train.Saver(ema_restore)
            saver = tf.train.Saver()

        else:
            CKPT_NODE = 0               #ckpt模型状态码,1表存在,0表不存在.
            global_step = tf.Variable(0, trainable=False)

            #定义滑动平均
            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

            #定义saver实例
            saver = tf.train.Saver()

        #定义损失函数
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cem = tf.reduce_mean(ce)
        loss = cem + tf.add_n(tf.get_collection('losses'))

        #定义衰减学习率
        learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_step,
                mnist.train.num_examples / BATCH_SIZE,
                LEARNING_RATE_DECAY,
                staircase = True)


        #定义训练过程
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

        #会话
        with tf.Session() as sess:
            if CKPT_NODE :
                saver.restore(sess, ckpt.model_checkpoint_path)

                for i in range (steps):
                    xs, ys = mnist.train.next_batch(BATCH_SIZE)
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                    if (i+global_step_c) % 1000 == 0:
                        print("After %d training steps, loss on training batch is %g." % (step, loss_value))
                        #保存模型
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step= global_step)


            else:

                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                for i in range(STEPS):
                    #随机从训练集中抽取 BATCH_SIZE 组数据
                    xs, ys = mnist.train.next_batch(BATCH_SIZE)
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                    if i % 1000 == 0:
                        print("After %d training steps, loss on training batch is %g." % (step, loss_value))
                        #保存模型
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step= global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()
