#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.functional as F

class Highway(nn.Module):
    """
    Class that map x_conv to an embedding vector
    """
    def __init__(self, word_embed_size):
        """
        Init the Highway module
        @param word_embed_size (int): Embedding size (dimensionality) for both the input (conv_output) and output
        """
        super(Highway, self).__init__()
        self.word_embed_size   = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size)

    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        """
        Take a mini batch of convolution output, compute
        :param x_conv (torch.Tensor), shape batch_size x word_embed_size
        :return: word_embedding (torch.Tensor), shape batch_size x word_embed_size
        """
        x_proj = torch.relu_(self.proj(x_conv))
        x_gate = torch.sigmoid(self.gate(x_conv))

        x = x_gate * x_proj + (1 - x_gate) * x_conv
        return x

### END YOUR CODE 

