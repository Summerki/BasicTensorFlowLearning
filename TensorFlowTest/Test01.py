# 测试TensorFlow

import tensorflow as tf

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

result = a + b

print(result)  # 输出 Tensor("add:0", shape=(2,), dtype=float32)

