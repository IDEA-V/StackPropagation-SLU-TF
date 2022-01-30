from re import S
import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, RNN

class Embedding(Layer):
    def __init__(self, shape, dropout):
        var_initializer = tf.initializers.GlorotUniform()
        self.embedding_encoder = tf.Variable(var_initializer(shape), name="embedding")
        self.dropout = dropout
        super(Embedding, self).__init__()

    def call(self, inputs):
        word_tensor = tf.nn.embedding_lookup(self.embedding_encoder, inputs)
        word_tensor = tf.nn.dropout(word_tensor, rate=self.dropout)
        return word_tensor

class qkvEmbedding(Layer):
    def __init__(self, shape):
        var_initializer = tf.initializers.GlorotUniform()
        self.q_m = tf.Variable(var_initializer(shape), name="query_dense")
        self.k_m = tf.Variable(var_initializer(shape), name="key_dense")
        self.v_m = tf.Variable(var_initializer(shape), name="value_dense")
        super(qkvEmbedding, self).__init__()

    def call(self, word_tensor):
        q = tf.einsum('ijk,kl->ijl', word_tensor, self.q_m)
        k = tf.einsum('ijk,kl->ijl', word_tensor, self.k_m)
        v = tf.einsum('ijk,kl->ijl', word_tensor, self.v_m)
        return q,k,v

class Predict(Layer):
    def __init__(self, dense):
        self.dense = dense
        super().__init__()
    
    def call(self, input):
        intent_pred = tf.einsum('ijk,kl->ijl', input, self.dense)
        return intent_pred

class ForceDecoder(Layer):
    def __init__(self, num_intents, embedding_dim, hidden_dim, batch_size, dropout):
        self.batch_size = batch_size
        self.dropout = dropout

        var_initializer = tf.initializers.GlorotUniform()
        intent_initializer=tf.random_normal_initializer()
        self.intent_embedding = tf.Variable(var_initializer([num_intents, embedding_dim]), name="intent_embedding")
        self.intent_init_tensor = tf.Variable(intent_initializer([1, 1, embedding_dim]), name="intent_init")

        self.force_intent_decoder_cell = LSTM(hidden_dim, return_sequences=True, dropout = dropout)
        
        super(ForceDecoder, self).__init__()
    
    def call(self, input, force_tensor):
        force_embed = tf.nn.embedding_lookup(self.intent_embedding, force_tensor[:, :-1])
        decoder_input = tf.concat(
            [input, tf.concat([tf.tile(self.intent_init_tensor, [self.batch_size, 1, 1]), force_embed], 1)], 2)
        decoder_input = tf.nn.dropout(decoder_input, rate=self.dropout)
        
        force_decoder = self.force_intent_decoder_cell(decoder_input)
        force_decoder = tf.nn.dropout(force_decoder, rate=self.dropout)
        return force_decoder

class Decoder(Layer):
    def __init__(self, non_force_decoder_cell, hidden_dim, embedding_dim, num, force_rate, args):
        self.non_force_decoder = RNN(non_force_decoder_cell, return_sequences=True)
        self.force_decoder = ForceDecoder(num, embedding_dim, hidden_dim, args.batch_size, args.dropout_rate)
        self.rate = force_rate
        super().__init__()

    def call(self, input_tensor, force_tensor):
        if force_tensor is not None:
            if tf.random.uniform([1]) > self.rate:
                return self.force_decoder(input_tensor, force_tensor)
            else:
                return self.non_force_decoder(input_tensor)
        else:
            return self.non_force_decoder(input_tensor)