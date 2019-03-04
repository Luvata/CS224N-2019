#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check_handmade.py: Handmade sanity checks for implementation problems

Usage:
    sanity_check_handmade.py generate
    sanity_check_handmade.py highway
    sanity_check_handmade.py cnn

Options:
    -h --help       Show this screen.
"""

from docopt import docopt

from cnn import CNN
from sanity_check import DummyVocab

import torch
import torch.nn as nn
import numpy as np
from nmt_model import NMT

from highway import Highway
from char_decoder import CharDecoder
from vocab import Vocab

BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0
NUM_FILTER = 4
KERNEl_LEN = 3
MAX_WORD_LEN=8

def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)

def question_1h_generate_data():
    """
    Unit test data generator for Highway
    """
    conv_input = np.random.rand(BATCH_SIZE, EMBED_SIZE)
    W_proj     = np.ones((EMBED_SIZE, EMBED_SIZE)) * 0.3
    b_proj     = np.ones(EMBED_SIZE) * 0.1

    W_gate     = np.ones((EMBED_SIZE, EMBED_SIZE)) * 0.3
    b_gate     = np.ones(EMBED_SIZE) * 0.1

    def relu(inpt):
        return np.maximum(inpt, 0)

    def sigmoid(inpt):
        return 1. / (1 + np.exp(-inpt))

    x_proj    = relu(conv_input.dot(W_proj) + b_proj)
    x_gate = sigmoid(conv_input.dot(W_gate) + b_gate)
    x_highway = x_gate * x_proj + (1 - x_gate) * conv_input

    np.save('sanity_check_handmade_data/highway_conv_input.npy', conv_input)
    np.save('sanity_check_handmade_data/highway_output.npy', x_highway)


def question_1h_sanity_check(highway):
    """
    Sanity check for highway.py, basic shape check and bias check
    """
    reinitialize_layers(highway) #
    # print("Running shape check")

    inpt = torch.from_numpy(np.load('sanity_check_handmade_data/highway_conv_input.npy').astype(np.float32))
    outp_expected = torch.from_numpy(np.load('sanity_check_handmade_data/highway_output.npy').astype(np.float32))

    with torch.no_grad():
        outp = highway(inpt)

    outp_expected_size = [BATCH_SIZE, EMBED_SIZE]

    assert (np.allclose(outp.numpy(), outp_expected.numpy())), \
        "Highway output is incorrect: it should be:\n {} but is:\n{}".format(outp_expected, outp)
    # print("Passed all tests :D")

def question_1g_generate_data():
    pass

def question_1g_sanity_check(CNN):
    pass

def main():
    """ Main func.
    """
    args = docopt(__doc__)
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize Highway
    highway = Highway(
        word_embed_size=EMBED_SIZE,
        dropout_rate=DROPOUT_RATE
    )

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    cnn = CNN(
        char_embed_size=EMBED_SIZE,
        num_filters=NUM_FILTER,
        max_word_length=MAX_WORD_LEN,
        KERNEl_LEN=KERNEl_LEN
    )

    if args['highway']:
        question_1h_sanity_check(highway)
    elif args['cnn']:
        question_1g_sanity_check(cnn)
    elif args['generate']:
        question_1h_generate_data()
        question_1g_generate_data()
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()