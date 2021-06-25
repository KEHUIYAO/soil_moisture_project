import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import Dataset
import torch.optim as optim
from typing import *
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout <= 0:
            return x


        max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        return x

class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.):
        super(BayesianLSTM, self).__init__()


        self.dropouti = dropouti
        self.dropoutw = dropoutw

        self.input_drop = VariationalDropout(dropouti)
        self.output_drop = VariationalDropout(dropouto)

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Feed forward network
        self.layernorm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Linear(hidden_dim, 1)



    def forward(self, x, y):
        """

        :param x: N x t x (input_dim-1)
        :param y: N x t x 1
        :return:N x 1
        """
        input = torch.cat((x, y), 2)
        input = self.input_drop(input)
        _, (output, _) = self.lstm(input)
        output = self.output_drop(output)
        output = self.ffn(output)
        return output


