import argparse
from dataset import DataProcessor
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input

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

init_args()
args = parser.parse_args()
data_processor = DataProcessor()
slots_loss, intents_loss = 0, 0
train_data = data_processor.get_data("train", batch_size=args.batch_size)

text = Input()
embedding_encoder = tf.Variable(name="embedding", shape=[args.num_words, args.word_embedding_dim])
word_tensor = tf.nn.embedding_lookup(embedding_encoder, text)
