import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]
print(X)
print(Y_)

# 定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))  # 喂入特征
y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 喂入标签
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数以及反向传播算法
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)


# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = (i*BATCH_SIZE)%32 + BATCH_SIZE
        # print(start)
        # print(end)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print('STEP:%d' % (i))
            print(sess.run(w1))
    print('Final w1 is : \n', sess.run(w1))

