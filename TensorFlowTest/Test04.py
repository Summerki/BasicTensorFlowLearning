# 用placeholder实现输入定义（sess.run()中喂入一组数据）的情况

import tensorflow as tf

# 定义输入和参数
x = tf.placeholder(tf.float32, shape=(1,2))
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))


# 定义前向传播的情况
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 实现对所有变量的初始化
    sess.run(init_op)
    print("y in Test04 is : \n", sess.run(y, feed_dict={x:[[0.7, 0.5]]}))   #  [[3.0904665]]

    # 查看随机生成的权重
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))


