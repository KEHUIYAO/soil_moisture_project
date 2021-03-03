import torch.nn as nn
import torch
import numpy as np



class LSTM_3(nn.Module):
    def __init__(self, feature_dim, static_dim, hidden_dim_lstm, hidden_dim_ffn):


        super(LSTM_3,self).__init__()
        # the input data for the lstm is the feature vector + last time's output
        input_dim_lstm = feature_dim + 1
        # use 1-layer lstm
        num_layer = 1
        self.rnn = nn.LSTM(input_dim_lstm, hidden_dim_lstm, num_layer)

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
        x = x.permute(1,0,2)
        mask = mask.permute(1,0,2)
        features = features.permute(1,0,2)
        static = static.permute(1,0,2)
        seq_len = x.size(0)
        cur_y_hat = 0

        # store the result
        y_hat = []

        for i in range(seq_len):

            if i == 0:
                cur_input = x[i:(i+1),:,:]
                cur_features = features[i:(i+1),:,:]

                # combine with features, and the input for the rnn need to have dimension (seq_len, batch_size, feature_dim + 1)
                cur_input = torch.cat([cur_input, cur_features],dim = 2)


                # input of rnn is of shape (seq_len, batch_size, feature_dim)
                cur_output, (cur_hidden_state, cur_cell_state) = self.rnn(cur_input)


            else:
                # at each time step, if the data is missing, e.g. mask = 0,
                # we will use last time step's output as the current input.
                # mask[0] will not be zero.
                if mask[i,0,0] == 0:
                    cur_input = cur_y_hat
                else:

                    # use teacher force training
                    if np.random.rand(1) < teacher_force_ratio:
                        # print(np.random.rand(1))

                        cur_input = x[i:(i+1),:,:]
                    else:
                        cur_input = cur_y_hat

                cur_features = features[i:(i+1),:,:]

                # combine with features
                cur_input = torch.cat([cur_input, cur_features], dim = 2)

                # LSTM's output has dimension (seq_len, batch_size, hidden_dim)
                cur_output, (cur_hidden_state, cur_cell_state) = self.rnn(cur_input,(cur_hidden_state,cur_cell_state))

            # ffn requires its input has dimension (batch_size, hidden_dim), and the output has dimension (batch_dim, output_dim)
            cur_output = cur_output.squeeze(0)
            static_input = static[i,:,:]
            ffn_input = torch.cat([cur_output, static_input], dim = 1)


            # predict the output at current time step, which also has  (seq_len, batch_size, 1)
            cur_y_hat = self.ffn(ffn_input).unsqueeze(0)

            y_hat.append(cur_y_hat)

        # stack on the time index, y_hat is of shape (seq_len, batch_size, 1)
        y_hat = torch.cat(y_hat, dim=0)

        # permute y_hat to be of shape (batch_size, seq_len, 1)
        y_hat = y_hat.permute(1,0,2)
        return y_hat

