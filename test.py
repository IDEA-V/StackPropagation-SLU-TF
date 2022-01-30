from random import random
import tensorflow as tf
from tensorflow.keras import Input, Model

text = Input(shape=(None,), dtype=tf.float32, batch_size=2)


class Embedding(tf.keras.layers.Layer):
    def __init__(self, shape, dropout):
        super(Embedding, self).__init__()
        var_initializer = tf.initializers.GlorotUniform()
        self.shape = shape
        self.embedding_encoder = tf.Variable(var_initializer(shape), name="embedding")
        self.dropout = dropout

    def call(self, inputs):
        word_tensor = tf.nn.embedding_lookup(self.embedding_encoder, inputs)
        word_tensor = tf.nn.dropout(word_tensor, rate=self.dropout)
        return word_tensor

class Add(tf.keras.layers.Layer):

    def __init__(self):
        var_initializer = tf.initializers.GlorotUniform()
        self.a = tf.Variable([1.,1.], name="embedding")
        super().__init__()
    
    def call(self, inputs, a):
        print(a)
        word_tensor = inputs + self.a
        word_tensor = word_tensor
        return word_tensor

a = tf.keras.layers.Lambda(lambda x: [random(), random()])
num = a(text)
word_emb = Add()
word_tensor = word_emb(text, num)


train_model = Model(inputs=[text], outputs=[word_tensor])

print("================================")
train_model.compile(run_eagerly = True, loss="mean_squared_error", optimizer="Adam")
train_model.fit([1.1,2.1,3.1,4.1],[11,21,31,41], batch_size=2)