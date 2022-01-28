import argparse
from io import BufferedWriter
import numpy as np
import math
from random import random

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Bidirectional, RNN

from dataset import DataProcessor

parser = argparse.ArgumentParser()

def init_args():
    parser.add_argument('--data_dir', '-dd', type=str, default='data/snips')
    parser.add_argument('--save_dir', '-sd', type=str, default='save')
    parser.add_argument("--random_state", '-rs', type=int, default=0)
    parser.add_argument('--num_epoch', '-ne', type=int, default=300)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
    parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
    parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
    parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)
    parser.add_argument("--differentiable", "-d", action="store_true", default=False)
    parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

    # model parameters.
    parser.add_argument('--num_words', '-nw', type=int, default=12150)
    parser.add_argument('--num_intents', '-ni', type=int, default=8)
    parser.add_argument('--num_slots', '-ns', type=int, default=73)
    parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
    parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
    parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
    parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
    parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
    parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
    parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
    parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)

def loop_fn_build(inputs, init_tensor, embedding_matrix, cell, batch_size, embedding_size, sequence_length, dense,
                  sentence_size):

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # [B,T,S]-->[T,B,S]
    inputs_trans = tf.transpose(inputs, perm=[1, 0, 2])
    inputs_ta = inputs_ta.unstack(inputs_trans)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            prev_intent = tf.squeeze(tf.tile(init_tensor, [1, batch_size, 1]), [0])
        else:
            next_cell_state = cell_state
            dense_layer = tf.einsum('jk,kl->jl', cell_output, dense)
            _, index = tf.math.top_k(dense_layer, k=1)
            prev_intent = tf.nn.embedding_lookup(embedding_matrix, tf.reshape(index, [-1]))
        elements_finished = (time >= sequence_length)
        finished = tf.reduce_all(elements_finished)
        next_input = tf.cond(finished,
                             lambda: tf.zeros([batch_size, embedding_size], dtype=tf.float32),
                             lambda: tf.concat([inputs_ta.read(time), prev_intent], 1))
        # next_input =  tf.concat([inputs_ta.read(time), prev_intent], 1)

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    return loop_fn


def build_decoder(input_tensor, force_tensor, cond_tensor, embedding_tensor, init_tensor, batch_size, keep_prob,
                  decoder_cell, args, dense, sentence_size, sequence_length, embedding_dim):
    def force():
        force_embed = tf.nn.embedding_lookup(embedding_tensor, force_tensor[:, :-1])
        decoder_input = tf.concat(
            [input_tensor, tf.concat([tf.tile(init_tensor, [batch_size, 1, 1]), force_embed], 1)], 2)
        decoder_input = tf.nn.dropout(decoder_input, rate=1-keep_prob)
        
        force_decoder = decoder_cell(decoder_input)
        force_decoder = tf.nn.dropout(force_decoder, rate=1-keep_prob)

        return force_decoder

    def non_force():
        outputs, final_state, _ = tf.compat.v1.nn.raw_rnn(
            decoder_cell,
            loop_fn_build(input_tensor,
                          init_tensor,
                          embedding_tensor,
                          decoder_cell,
                          batch_size,
                          embedding_dim + args.encoder_hidden_dim + args.attention_output_dim,
                          sequence_length,
                          dense,
                          sentence_size
                          )
        )
        non_force_decoder = tf.transpose(outputs.stack(), [1, 0, 2])
        return non_force_decoder

    a = force()
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # [B,T,S]-->[T,B,S]
    inputs_trans = tf.transpose(input_tensor, perm=[1, 0, 2])
    inputs_ta = inputs_ta.unstack(inputs_trans)

    if force_tensor is not None:
        # decoder = force()
        decoder = tf.cond(cond_tensor, lambda: force(), lambda: non_force(), name="Condition_Input_Feeding")
    else:
        decoder = non_force()
    return decoder


init_args()
args = parser.parse_args()
data_processor = DataProcessor()
slots_loss, intents_loss = 0, 0
train_data = data_processor.get_data("train", batch_size=args.batch_size)

text = Input(shape=(None,), dtype=tf.int32)
slots = Input(shape=(None,), dtype=tf.int32)
word_weights = Input(shape=(None,), dtype=tf.float32)
sequence_length = Input(shape=(None,), dtype=tf.int32)
intent = Input(shape=(None,), dtype=tf.int32)
intent_feeding = random() > args.intent_forcing_rate
slot_feeding = random() > args.slot_forcing_rate

batch_size, sentence_size = tf.shape(text)[0], tf.shape(text)[1]

print("=============================")
print(batch_size)
print(sentence_size)

slots_shape = tf.shape(slots)
var_initializer = tf.initializers.GlorotUniform()

embedding_encoder = tf.Variable(var_initializer([args.num_words, args.word_embedding_dim]), name="embedding")

word_tensor = tf.nn.embedding_lookup(embedding_encoder, text)
word_tensor = tf.nn.dropout(word_tensor, rate=args.dropout_rate)


# lstm encoder
lstm_fw = LSTM(args.encoder_hidden_dim // 2, return_sequences=True, dropout = args.dropout_rate)
lstm_bw = LSTM(args.encoder_hidden_dim // 2, return_sequences=True, dropout = args.dropout_rate, go_backwards = True)

lstm_encoder = Bidirectional(lstm_fw, backward_layer = lstm_bw, merge_mode=None)
lstm_encoder = lstm_encoder(word_tensor)

fw = tf.nn.dropout(lstm_encoder[0], rate=args.dropout_rate)
bw = tf.nn.dropout(lstm_encoder[1], rate=args.dropout_rate)

lstm_encoder = tf.concat([fw, bw], 2)

#self attention
q_m = tf.Variable(var_initializer([args.word_embedding_dim, args.attention_hidden_dim]), name="query_dense")
k_m = tf.Variable(var_initializer([args.word_embedding_dim, args.attention_hidden_dim]), name="key_dense")
v_m = tf.Variable(var_initializer([args.word_embedding_dim, args.attention_output_dim]), name="value_dense")
q = tf.einsum('ijk,kl->ijl', word_tensor, q_m)
k = tf.einsum('ijk,kl->ijl', word_tensor, k_m)
v = tf.einsum('ijk,kl->ijl', word_tensor, v_m)
score = tf.nn.softmax(tf.matmul(q, k, transpose_b=True)) / math.sqrt(args.attention_hidden_dim)
attention_tensor = tf.matmul(score, v)
attention_tensor = tf.nn.dropout(attention_tensor, rate=args.dropout_rate)

encoder = tf.concat([lstm_encoder, attention_tensor], 2)

initializer=tf.random_normal_initializer()
#intent decoder
embedding_intent_decoder = tf.Variable(var_initializer([args.num_intents, args.intent_embedding_dim]), name="intent_embedding")
intent_init_tensor = tf.Variable(initializer([1, 1, args.intent_embedding_dim]), name="intent_init")

intent_decoder_cell = LSTM(args.slot_decoder_hidden_dim, return_sequences=True, dropout = args.dropout_rate)

intent_dense = tf.Variable(var_initializer([args.intent_decoder_hidden_dim, args.num_intents]), name="intent_dense")
intent_decoder = build_decoder(encoder, intent, intent_feeding, embedding_intent_decoder,
                                           intent_init_tensor, batch_size, 1-args.dropout_rate, intent_decoder_cell,
                                           args, intent_dense, sentence_size, sequence_length,
                                           args.intent_embedding_dim)
intent_pred = tf.einsum('ijk,kl->ijl', intent_decoder, intent_dense)
slot_combine = tf.concat([encoder, intent_pred], 2, name='slotInput')

#slot decoder
embedding_slot_decoder = tf.Variable(var_initializer([args.num_slots, args.slot_embedding_dim]), name="slot_embedding")
slot_init_tensor = tf.Variable(initializer([1, 1, args.slot_embedding_dim]), name ="slot_init", shape=[1, 1, args.slot_embedding_dim])

slot_decoder_cell = LSTMCell(args.slot_decoder_hidden_dim)
slot_decoder_cell = tf.nn.RNNCellDropoutWrapper(slot_decoder_cell, input_keep_prob=args.dropout_rate,
                                 output_keep_prob=args.dropout_rate)
slot_dense = tf.Variable(name="slot_dense", shape=[args.slot_decoder_hidden_dim, args.num_slots])
slot_decoder = build_decoder(slot_combine, slots, slot_feeding, embedding_slot_decoder,
                                         slot_init_tensor, batch_size, args.dropout_rate, slot_decoder_cell,
                                         args, slot_dense, sentence_size, sequence_length,
                                         args.slot_embedding_dim + args.num_intents)
slot_pred = tf.einsum('ijk,kl->ijl', slot_decoder, slot_dense)


train_model = Model(inputs=[text, slots, word_weights, sequence_length, intent], outputs=[slot_pred])

#slot loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots, logits=slot_pred)
cross_entropy = tf.reshape(cross_entropy, slots_shape)
slot_loss = tf.reduce_sum(cross_entropy * word_weights, 1)

#intent loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_pred)
cross_entropy = tf.reshape(cross_entropy, slots_shape)
intent_loss = tf.reduce_sum(cross_entropy * word_weights, 1)

loss = intent_loss + slot_loss
opt = tf.optimizers.Adam()