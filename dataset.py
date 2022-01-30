import random
import math
import numpy as np
from tensorflow.keras.utils import Sequence

def create_vocab(name):
    path = "./alphabet/" + name
    file = open(path + "_dict.txt", mode="r", encoding="utf8")
    lines = file.readlines()
    list = []
    dict = {}
    for i in lines:
        value, index = i.split("\t")
        list.append(value)
        dict[value] = int(index)
    return list, dict


def get_items(path, word_dict, intent_dict, slot_dict):
    file = open(path, mode="r", encoding="utf8")
    item = []
    lines = file.readlines()
    tmp_word = []
    tmp_slot = []
    for i in lines:
        if i == "\n":
            continue
        i = i.replace("\n", "", -1)
        line = i.split(" ")
        if (len(line)) == 1:
            item.append((tmp_word, [1 for j in range(len(tmp_word))], tmp_slot,
                         [intent_dict[line[0]] for j in range(len(tmp_word))]))
            tmp_slot = []
            tmp_word = []
        else:
            if line[0] in word_dict:
                tmp_word.append(word_dict[line[0]])
            else:
                tmp_word.append(1)
            tmp_slot.append(slot_dict[line[1]])
    return item


class DataProcessor(Sequence):
    def __init__(self, batch_size):
        self.intent_list, self.intent_dict = create_vocab("intent")
        self.word_list, self.word_dict = create_vocab("word")
        self.slot_list, self.slot_dict = create_vocab("slot")
        self.batch_size = batch_size
        self.data = {'train': get_items(
            "./data/snips/train.txt",
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        ), 'test': get_items(
            "./data/snips/test.txt",
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        ), 'dev': get_items(
            "./data/snips/dev.txt",
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        )}
        self.choose_data("train")

    def choose_data(self, name):
        self.name = name
        data = self.data[self.name]
        random.shuffle(data)
        self.word, self.seq, self.slot, self.intent = ([data[i][j] for i in range(len(data))] for j in range(4))

    def __len__(self):
        return math.ceil(len(self.data[self.name]) / self.batch_size)

    def __getitem__(self, i):
        batch = (self.word[i:i + self.batch_size], self.seq[i:i + self.batch_size], self.slot[i:i + self.batch_size], self.intent[i:i + self.batch_size])
        max_len = max([len(item) for item in batch[0]])
        length = [[len(item)]*max_len for item in batch[0]]

        x = [[item + [0 for i in range(max_len - len(item))] for item in batch[j]] for j in range(len(batch))]+[length]

        for i in range(len(x)):
            x[i] = np.asarray(x[i]).astype(np.int32)

        y = x[2:4]
        z = x[1]
        x = [x[0]] + x[2:]
        return x, y, z



if __name__ == "__main__":
    # datasets = get_dataset(8, 73, 16, 10)
    data_processor = DataProcessor(16)
    a = data_processor[0]
    b = 1