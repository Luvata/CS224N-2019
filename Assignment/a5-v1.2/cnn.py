#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size=5 ):
        """
        Init the character CNN module
        :param char_embed_size (int): embedding size (dimensionality) of each character in a word
        :param num_filters (int): a.k.a number of output channels or word_embed_size
        :param max_word_length (int): maximum length of a word
        :param kernel_size (int): length convolve of each kernel
        """
        # input  : batch, char_embed  , max_word_length
        # output : batch, num_filters , convolution_length
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size # num input channels
        self.num_filters     = num_filters     # num output channels
        self.kernel_size     = kernel_size
        self.max_word_length = max_word_length

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            bias=True
        )

        self.max_pool_1d = nn.MaxPool1d(max_word_length - kernel_size + 1)
        # self.conv1d.w

    def forward(self, input):
        """
        Take a mini batch of character embedding of each word, compute word embedding
        :param input (Tensor): shape (batch_size, char_embed_size, max_word_length)
        :return (Tensor): shape (batch_size, word_embed_size), word embedding of each word in batch
        """
        x = self.conv1d(input) # (batch_size, word_embed_size, max_word_length - kernel_size + 1)
        x = self.max_pool_1d(F.relu_(x)).squeeze()  # (batch_size, word_embed_size)
        return x


### END YOUR CODE

