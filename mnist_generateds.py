#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = 'mnist_data_jpg/mnist_train_jpg_60000/'      #训练图片路径
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt' #训练图片标签路径
tfRecord_train = './data/mnist_train.tfrecords'             
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'      #测试图片路径
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt' #测试图片标签路径
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'    #数据路径
resize_height = 28
resize_width = 28



def write_tfRecord(tfRecordName, image_path, label_path):
    #新建一个writer
    with tf.python_io.TFRecordWriter(tfRecordName) as writer:
        num_pic = 0     #进度数
        with open(label_path, 'r') as f:
            contents = f.readlines()
        for content in contents:
            value = content.split()
            img_path = image_path + value[0]
            img = Image.open(img_path)
            img_raw = img.tobytes()     #将图片转化为str格式
            labels = [0] * 10
            labels[int(value[1])] = 1    #创建标签数列
            #把每张图片和标签封装到example中
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': 
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': 
                tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }))
            #把example进行序列化
            writer.write(example.SerializeToString())
            num_pic += 1
            print ("the number of picture:", num_pic)
    print("write tfrecord successful")


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()    #新建一个reader
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
                                        features={ 
                                            #解序列化时键名应与创建时的键名相同
                                            #标签分类指明是10分类
                                            'label':tf.FixedLenFeature([10], tf.int64), 
                                            'img_raw': tf.FixedLenFeature([], tf.string)
                                            })
    img = tf.decode_raw(features['img_raw'], tf.uint8) #将字符串转换为无符号8位整形
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print 'The directory was created successfully'
    else:
        print 'directory already exists'
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def get_tfRecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], 
                                                    batch_size = num, #打乱输出 batch_size 组数据
                                                    num_threads = 2, #整个过程使用了两个线程
                                                    capacity = 1000, #从总样本中顺序取出 capacity 组数据
    #如果 capacity 小于 mni_after_dequeue 值,将从总样本中取数据填满 capacity
                                                    mni_after_dequeue = 700)

    #返回 num组随机 数据
    return img_batch, label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()

