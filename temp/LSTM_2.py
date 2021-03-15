import torch.nn as nn
import torch
import numpy as np



class LSTM_2(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layer, bias):
        """
        lstm with features

        :param hidden_dim: number of dimensions of the hidden state.
        :param num_layer: number of stacked LSTM layers
        :param bias: whether or not to add the bias
        """

        super(LSTM_2,self).__init__()
        # input feature is a scalar
        self.input_dim = input_dim

        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layer)
        self.ffn = nn.Linear(hidden_dim, 1, bias)
        # self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
        #                          nn.ReLU(),
        #                          nn.Linear(hidden_dim, 1)
        #                          )


    def forward(self, x, mask, features, teacher_force_ratio):
        """

        :param
        x: x is a tensor with dimension seq_len.
        mask:  mask is a tensor with dimension seq_len.
        features: features is a tensor with dimension (seq_len, num_of_features)

        :return:
        """
        seq_len = x.size(0)
        cur_y_hat = 0

        # store the result
        y_hat = []

        for i in range(seq_len):

            if i == 0:
                cur_input = x[i].view(1,1,1)
                cur_features = features[i, :].view(1, 1, -1)

                # combine with features, and the input for the rnn need to have dimension (seq_len, batch_size, feature_dim + 1)
                cur_input = torch.cat([cur_input, cur_features],dim = 2)


                cur_output, (cur_hidden_state, cur_cell_state) = self.rnn(cur_input)


            else:
                # at each time step, if the data is missing, e.g. mask = 0,
                # we will use last time step's output as the current input.
                # mask[0] will not be zero.
                if mask[i] == 0:
                    cur_input = cur_y_hat
                else:

                    # use teacher force training
                    if np.random.rand(1) < teacher_force_ratio:
                        # print(np.random.rand(1))
                        # x[i] is a scalar
                        cur_input = x[i].view(1,1,1)
                    else:
                        cur_input = cur_y_hat

                cur_features = features[i, :].view(1, 1, -1)

                # combine with features
                cur_input = torch.cat([cur_input, cur_features], dim = 2)

                # LSTM's output has dimension (seq_len, batch_size, hidden_dim)
                cur_output, (cur_hidden_state, cur_cell_state) = self.rnn(cur_input,(cur_hidden_state,cur_cell_state))

            # ffn requires its input has dimension (batch_size, hidden_dim), and the output has dimension (batch_dim, output_dim)
            cur_y_hat = self.ffn(cur_output.view(1, self.hidden_dim)).view(1,1,1)
            y_hat.append(cur_y_hat)

        # stack on the time index and flatten
        y_hat = torch.stack(y_hat,0).view(-1)

        return y_hat

