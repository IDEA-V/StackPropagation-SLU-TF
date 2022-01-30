import tensorflow as tf
from tensorflow.keras.layers import RNN, LSTMCell, Layer

class myLSTMCell(Layer):
    
    def __init__(self, hidden_dim, emb_dim, num_intents):
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.state_size = [hidden_dim, hidden_dim, emb_dim]
        self.lstm = LSTMCell(hidden_dim)
        var_initializer = tf.initializers.GlorotUniform()
        self.dense = tf.Variable(var_initializer([hidden_dim, num_intents]), name="intent_dense")
        self.embedding_matrix= tf.Variable(var_initializer([num_intents, emb_dim]), name="intent_embedding")
        super().__init__()

    def call(self, input, state):
        concatInput = tf.concat([input, state[2]], 1)
        output, newState = self.lstm(concatInput, state[:2])
        
        dense_layer = tf.einsum('jk,kl->jl', output, self.dense)
        _, index = tf.math.top_k(dense_layer, k=1)
        prev_intent = tf.nn.embedding_lookup(self.embedding_matrix, tf.reshape(index, [-1]))

        newState.append(prev_intent)

        return output, newState
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        states = self.lstm.get_initial_state(inputs, batch_size, dtype)
        states.append(tf.zeros((batch_size, self.emb_dim)))
        return states
    