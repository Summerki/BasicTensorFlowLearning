import tensorflow as tf

x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])

print(x.shape)  # (1, 2)
print(w.shape)  # (2, 1)

y = tf.matmul(x, w)
print(y)  # 输出  Tensor("MatMul:0", shape=(1, 1), dtype=float32)

with tf.Session() as sess:
    print(sess.run(y))  # 输出 [[11.]]