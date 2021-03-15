import torch.nn as nn
import torch
import numpy as np

import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


# soil moisture gap filling model

class SoilMoistureGapFilling(nn.Module):
    def __init__(self, time_varying_dim, static_dim, lstm_hidden_dim, ffn_hidden_dim, num_layers=1, bidirectional=False, bias=True, dropout = 0, direct_connection_from_previous_output = True):
        super(SoilMoistureGapFilling,self).__init__()
        # if there is a direct connection from previous output to current hidden state
        if direct_connection_from_previous_output:
            print("use direct connection from previous output!")
            lstm_input_dim = time_varying_dim + 1
        else:
            print("do not use direct connection from previous output!")
            lstm_input_dim = time_varying_dim

        self.direct_connection_from_previous_output = direct_connection_from_previous_output

        # LSTM
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, num_layers=num_layers, bias=True, dropout = dropout, bidirectional=bidirectional)


        # Feed forward network
        self.ffn = nn.Sequential(nn.Linear(lstm_hidden_dim + static_dim, ffn_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(ffn_hidden_dim, 1)
                                 )



    def forward(self, x, mask, time_varying_features, static_features, teacher_force_ratio):

        # since RNN receives the input with shape (seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)
        mask = mask.permute(1, 0, 2)
        features = time_varying_features.permute(1, 0, 2)
        static = static_features.permute(1, 0, 2)
        seq_len = x.size(0)


        if self.direct_connection_from_previous_output:
            cur_y_hat = 0
            # store the result
            y_hat = []

            for i in range(seq_len):
                if i == 0:
                    cur_input = x[i:(i + 1), :, :]
                    cur_features = features[i:(i + 1), :, :]
                    # combine with time-varying features, and the input for the rnn need to have dimension (seq_len, batch_size, feature_dim + 1)
                    cur_input = torch.cat([cur_input, cur_features], dim=2)
                    # input of rnn is of shape (seq_len, batch_size, feature_dim)
                    cur_output, (cur_hidden_state, cur_cell_state) = self.lstm(cur_input)
                else:
                    # at each time step, if the data is missing, e.g. mask = 0,
                    # we will use last time step's output as the current input.
                    # mask[0] will not be zero.
                    if mask[i, 0, 0] == 0:
                        cur_input = cur_y_hat
                    else:
                        # use teacher force training
                        if np.random.rand(1) < teacher_force_ratio:
                            # print(np.random.rand(1))
                            cur_input = x[i:(i + 1), :, :]
                        else:
                            cur_input = cur_y_hat

                    cur_features = features[i:(i + 1), :, :]
                    # combine with features
                    cur_input = torch.cat([cur_input, cur_features], dim=2)
                    # LSTM's output has dimension (seq_len, batch_size, hidden_dim)
                    cur_output, (cur_hidden_state, cur_cell_state) = self.lstm(cur_input,
                                                                              (cur_hidden_state, cur_cell_state))

                # ffn requires its input has dimension (batch_size, input_dim), and the output has dimension (batch_dim, output_dim)
                cur_output = cur_output.squeeze(0)
                static_input = static[i, :, :]
                ffn_input = torch.cat([cur_output, static_input], dim=1)

                # predict the output at current time step, which also has  (seq_len, batch_size, 1)
                cur_y_hat = self.ffn(ffn_input).unsqueeze(0)
                y_hat.append(cur_y_hat)

            # stack on the time index, y_hat is of shape (seq_len, batch_size, 1)
            y_hat = torch.cat(y_hat, dim=0)

            # permute y_hat to be of shape (batch_size, seq_len, 1)
            y_hat = y_hat.permute(1, 0, 2)


        else:
            # lstm's input has (seq_len, batch_size, feature_dim)
            # lstm's output has (seq_len, batch_size, hidden_dim_lstm)
            output, (hidden_state, cell_state) = self.lstm(features)

            # ffn requires its input has dimension (batch_size, hidden_dim), and the output has dimension (batch_dim, output_dim)
            y_hat = []
            for i in range(seq_len):
                cur_output = output[i, :, :]
                static_input = static[i, :, :]
                ffn_input = torch.cat([cur_output, static_input], dim=1)
                # predict the output at current time step, which also has (seq_len, batch_size, 1)
                cur_y_hat = self.ffn(ffn_input).unsqueeze(0)
                y_hat.append(cur_y_hat)

            # stack on the time index
            y_hat = torch.cat(y_hat, dim=0)

            # permute y_hat of shape (batch_size, seq_len, 1)
            y_hat = y_hat.permute(1, 0, 2)

        return y_hat






