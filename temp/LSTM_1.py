import torch.nn as nn
import torch
import numpy as np

class LSTM_1(nn.Module):
    def __init__(self,hidden_dim,num_layer, bias):
        """
        lstm without covariates

        :param hidden_dim: number of dimensions of the hidden state.
        :param num_layer: number of stacked LSTM layers
        :param bias: whether or not to add the bias
        """

        super(LSTM_1,self).__init__()
        # input feature is a scalar
        self.input_dim = 1

        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layer)
        self.ffn = nn.Linear(hidden_dim,self.input_dim,bias)


    def forward(self, x, mask, teacher_force_ratio):
        """

        :param
        x: x is a tensor with dimension seq_len.
        mask:  mask is a tensor with dimension seq_len.

        :return:
        """
        seq_len = x.size(0)
        cur_y_hat = 0
        # store the result
        y_hat = []

        for i in range(seq_len):




            # at time step 0, the hidden state and the cell state is initialized to be 0
            if i == 0:
                cur_input = x[i].view(1,1,1)
                cur_output, (cur_hidden_state, cur_cell_state) = self.rnn(cur_input)

            # at time step > 1, if the data is missing, e.g. mask = 0,
            # we will use last time step's output as the current input.
            # mask[0] will not be zero.
            else:
                if mask[i] == 0:
                    cur_input = cur_y_hat
                else:
                    # use teacher force ratio
                    if np.random.rand(1) < teacher_force_ratio:

                        # lstm's input should has dimension (seq_len, 1, input_features), x[i] is a scalar
                        cur_input = x[i].view(1,1,1)
                    else:
                        cur_input = cur_y_hat

                # cur_output has dimension (seq_len, 1, hidden_size), hidden state and cell state is updated for each time.
                cur_output, (cur_hidden_state, cur_cell_state) = self.rnn(cur_input,(cur_hidden_state,cur_cell_state))

            # ffn requires its input has dimension (1,hidden_size), and the output has dimension (1,1)
            cur_y_hat = self.ffn(cur_output.view(1, self.hidden_dim)).view(1, 1, 1)
            y_hat.append(cur_y_hat)

        # stack on the time index
        y_hat = torch.stack(y_hat,0)
        y_hat = y_hat.view(-1)
        return y_hat

