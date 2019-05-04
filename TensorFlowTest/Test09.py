import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 查看指定路径是否有数据集
mnist = input_data.read_data_sets(r'./data/',one_hot=True)

print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)

print(mnist.train.labels[0])

print(mnist.train.images[0])


x = [[1., 1.], [2., 2.]]
print(tf.reduce_mean(x))
print(tf.reduce_mean(x, 0))
print(tf.reduce_mean(x, 1))

with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(x, 1)))