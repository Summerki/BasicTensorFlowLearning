import tensorflow as tf
import numpy as np

BATCH_SIZE = 8  # 一次喂入NN多少组数据
seed = 23455

# 产生基于seed的随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵，表示32组输入数据集
X = rng.rand(32, 2)
print(X)
# 从X特征矩阵中取出一行，判断如果之和小于1给Label Y赋值1，若不小于1赋值0
# 将Y作为标签
Y = []
for i in range(X.shape[0]):
    if np.sum(X[i]) < 1:
        Y.append([1])
    else:
        Y.append([0])

# Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print(Y)

# 为了验证numpy中axis  0为竖轴   1代表横轴
# for i in range(X.shape[0]):
#     if(i == 0):
#         print(X[i])
#         np.reshape(X[i], (1,2))
#         a = np.array(X[i])
#         b = a.reshape((1,2))
#         print(b)
#         print(np.sum(b, axis=0))
#         print(np.sum(b, axis=1))

# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))  # 特征占位
y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 标签占位

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y-y_))  # tf.square：对每个元素求平方
                                        # 那tf.reduce_mean:将元素相加后除以元素个数n
# 反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # 随机梯度下降  学习率为0.001


# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前未经训练的参数取值
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
    print('\n')

    # 训练模型
    STEPS = 3000  # 迭代3000次
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        # print('start: %d ---> end: %d'%(start, end))
        # print(start)
        # print(end)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After %d training steps, loss on all data is %g'%(i, total_loss))

    # 输出训练后的参数取值
    print('\n')
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
