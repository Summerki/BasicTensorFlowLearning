import tensorflow as tf

# 设损失函数为(w+1)^2,令w初值为5，利用梯度下降求最小loss对应的w值

w = tf.Variable(tf.constant(5, dtype=tf.float32))
loss = tf.square(w+1)
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)  #可以理解为每进行一次run(Train_step)就是进行一次反向传播
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('After %s steps: w is %f, loss is %f'%(i, w_val, loss_val))
