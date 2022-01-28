import tensorflow as tf

input_data = tf.compat.v1.placeholder(tf.int32, [None, None], name='inputs')

a = tf.shape(input_data)

print(a[0])