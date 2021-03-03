import torch.nn as nn
import torch
import numpy as np



class LSTM_4(nn.Module):
    def __init__(self, feature_dim, static_dim, hidden_dim_lstm, hidden_dim_ffn, bidirectional = False):


        super(LSTM_4,self).__init__()

        # use 1-layer lstm
        num_layer = 1
        self.rnn = nn.LSTM(feature_dim, hidden_dim_lstm, num_layer, bidirectional)

        #self.ffn = nn.Linear(hidden_dim + static_dim, 1, bias)
        # combine the static features with lstm's hidden layer and feed them to a ffn
        self.ffn = nn.Sequential(nn.Linear(hidden_dim_lstm+static_dim, hidden_dim_ffn),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim_ffn, 1)
                                 )


    def forward(self, x, mask, features, static, teacher_force_ratio):
        """

        :param x: (batch, seq_len, 1)
        :param mask: (batch, seq_len, 1)
        :param features: (batch, seq_len, 5)
        :param static: (batch, seq_len, 10)
        :param teacher_force_ratio: teacher force ratio for training
        :return: predicted y, (batch, seq_len, 1)
        """

        # since RNN receives the input with shape (seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)
        mask = mask.permute(1, 0, 2)
        features = features.permute(1, 0, 2)
        static = static.permute(1, 0, 2)
        seq_len = x.size(0)



        # lstm's input has (seq_len, batch_size, feature_dim)
        # lstm's output has (seq_len, batch_size, hidden_dim_lstm)
        output, (hidden_state, cell_state) = self.rnn(features)


        # ffn requires its input has dimension (batch_size, hidden_dim), and the output has dimension (batch_dim, output_dim)
        y_hat = []
        for i in range(seq_len):
            cur_output = output[i,:,:]
            static_input = static[i,:,:]
            ffn_input = torch.cat([cur_output, static_input], dim = 1)
            # predict the output at current time step, which also has (seq_len, batch_size, 1)
            cur_y_hat = self.ffn(ffn_input).unsqueeze(0)
            y_hat.append(cur_y_hat)

        # stack on the time index
        y_hat = torch.cat(y_hat, dim=0)

        # permute y_hat of shape (batch_size, seq_len, 1)
        y_hat = y_hat.permute(1,0,2)

        return y_hat

