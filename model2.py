import argparse
from io import BufferedWriter
import numpy as np
import math
from random import random

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Bidirectional, RNN, Embedding, Lambda
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy

from dataset import DataProcessor
from myLSTM import myLSTMCell
from customLayers import qkvEmbedding, ForceDecoder, Predict, Decoder

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


def build_decoder(input_tensor, force_tensor, cond_tensor, hidden_dim, embedding_dim, num, args, non_force_decoder_cell):
    
    force_decoder = ForceDecoder(num, embedding_dim, hidden_dim, args.batch_size, args.dropout_rate) 
    
    def force(force_decoder):
        force_decoder = force_decoder(input_tensor, force_tensor)
        return force_decoder

    non_force_decoder = RNN(non_force_decoder_cell, return_sequences=True)
    def non_force(non_force_decoder):
        non_force_decoder = non_force_decoder(input_tensor)
        
        return non_force_decoder

    if force_tensor is not None:
        decoder = tf.cond(cond_tensor, lambda: force(force_decoder), lambda: non_force(non_force_decoder), name="Condition_Input_Feeding")
    else:
        decoder = non_force()
    return decoder


init_args()
args = parser.parse_args()

text = Input(shape=(None,), dtype=tf.int32, batch_size=args.batch_size)
slots = Input(shape=(None,), dtype=tf.int32, batch_size=args.batch_size)
intent = Input(shape=(None,), dtype=tf.int32, batch_size=args.batch_size)
sequence_length = Input(shape=(None,), dtype=tf.int32, batch_size=args.batch_size)

intent_feeding = Lambda(lambda x: random() > args.intent_forcing_rate)
intent_feeding = intent_feeding(text)
slot_feeding = Lambda(lambda x: random() > args.slot_forcing_rate)
slot_feeding = slot_feeding(text)

batch_size, sentence_size = tf.shape(text)[0], tf.shape(text)[1]

slots_shape = tf.shape(slots)

word_embedding = Embedding(args.num_words, args.word_embedding_dim)
word_tensor = word_embedding(text)
word_tensor = tf.nn.dropout(word_tensor, rate=args.dropout_rate)

# lstm encoder
lstm_fw = LSTM(args.encoder_hidden_dim // 2, return_sequences=True, dropout = args.dropout_rate)
lstm_bw = LSTM(args.encoder_hidden_dim // 2, return_sequences=True, dropout = args.dropout_rate, go_backwards = True)

lstm_encoder = Bidirectional(lstm_fw, backward_layer = lstm_bw, merge_mode=None)
lstm_encoder = lstm_encoder(word_tensor)

fw = tf.nn.dropout(lstm_encoder[0], rate=args.dropout_rate)
bw = tf.nn.dropout(lstm_encoder[1], rate=args.dropout_rate)

lstm_encoder = tf.concat([fw, bw], 2)

qkv = qkvEmbedding([args.word_embedding_dim, args.attention_hidden_dim])
q,k,v = qkv(word_tensor)


score = tf.nn.softmax(tf.matmul(q, k, transpose_b=True)) / math.sqrt(args.attention_hidden_dim)
attention_tensor = tf.matmul(score, v)
attention_tensor = tf.nn.dropout(attention_tensor, rate=args.dropout_rate)

encoder = tf.concat([lstm_encoder, attention_tensor], 2)

#intent decoders

non_force_intent_decoder_cell = myLSTMCell(args.intent_decoder_hidden_dim, args.intent_embedding_dim, args.num_intents)
intent_decoder = Decoder(non_force_intent_decoder_cell, args.intent_decoder_hidden_dim, args.intent_embedding_dim, args.num_intents, args.intent_forcing_rate, args)
intent_pred = Predict(non_force_intent_decoder_cell.dense)

intent_decoder = intent_decoder(encoder, intent)
intent_pred = intent_pred(intent_decoder)

#slot decoder
slot_combine = tf.concat([encoder, intent_pred], 2, name='slotInput')

non_force_slot_decoder_cell = myLSTMCell(args.slot_decoder_hidden_dim, args.slot_embedding_dim + args.num_intents, args.num_slots)
slot_decoder = Decoder(non_force_slot_decoder_cell, args.slot_decoder_hidden_dim, args.slot_embedding_dim + args.num_intents, args.num_slots, args.slot_forcing_rate, args)
slot_pred = Predict(non_force_slot_decoder_cell.dense)

slot_decoder = slot_decoder(slot_combine, slots)
slot_pred = slot_pred(slot_decoder)

train_model = Model(inputs=[text, slots, intent, sequence_length], outputs=[slot_pred, intent_pred])

opt = tf.optimizers.Adam()
train_model.compile(opt, SparseCategoricalCrossentropy(from_logits=True), ['accuracy'])

train_data = DataProcessor(args.batch_size)
test_data = DataProcessor(args.batch_size)
test_data.choose_data("test")

train_model.fit(train_data, epochs=args.num_epoch, validation_data=test_data)