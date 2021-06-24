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
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LSTM(nn.LSTM):
    def __init__(self, input_dim, hidden_dim, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(input_dim, hidden_dim, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)


        # Feed forward network
        self.layernorm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        # for name, param in self.named_parameters():
        #     if "weight_hh" in name:
        #         nn.init.orthogonal_(param.data)
        #     elif "weight_ih" in name:
        #         nn.init.xavier_uniform_(param.data)
        #     elif "bias" in name and self.unit_forget_bias:
        #         nn.init.zeros_(param.data)
        #         param.data[self.hidden_size:2 * self.hidden_size] = 1
        # for name, param in self.named_parameters():
        #     #bound = 1/np.sqrt(self.hidden_size)
        #     #nn.init.uniform_(param.data, -bound, bound)
        #     nn.init.constant_(param.data, 0)
        # def weights_init_uniform_rule(m):
        #     classname = m.__class__.__name__
        #     # for every Linear layer in a model..
        #     if classname.find('Linear') != -1:
        #         # get the number of the inputs
        #         n = m.in_features
        #         y = 1.0 / np.sqrt(n)
        #         m.weight.data.uniform_(-y, y)
        #         m.bias.data.fill_(0)
        # self.apply(weights_init_uniform_rule)
        torch.nn.init.xavier_uniform(self.ffn.weight)


    def forward(self, input, hx=None):
        "return a tensor with size BATCH_SIZE x T"
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        lstm_output = self.output_drop(seq)
        lstm_output = self.layernorm(lstm_output)
        ffn_output = []
        for i in range(lstm_output.size(1)):
            ffn_output.append(self.ffn(lstm_output[:, i, :]).unsqueeze(2))
        ffn_output = torch.cat(ffn_output, dim=2).squeeze(1)
        return ffn_output



def train(model, device, dataLoader, optimizer, criterion, tail=5, teacher_force_ratio=0.5):
    model.to(device)
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataLoader):
        x, y = batch

        N = x.size(0)
        T = x.size(1)
        x = x.to(device)
        y = y.to(device)


        loss = 0
        # use dynamic programming here
        output_list = torch.zeros((N, tail)).double()

        output_list.to(device)


        for j in range(tail):
            xx = x[:, 1:(T - tail + j + 1), ]

            if j==0:
                yy = y[:, :(T-tail+j)].unsqueeze(2)
            else:
                yy = y[:, :(T - tail + j)].clone().detach()
                #yy =  y[:, :(T-tail+j)].clone().detach().to(device)


                for k in range(j):
                    if np.random.random() > teacher_force_ratio:
                        yy[:, (-j+k)] = output_list[:, k]

                yy = yy.unsqueeze(2)

            xx = torch.cat((xx,yy), 2)
            output = model(xx)
            output_list[:, j] = output[:,-1]


        loss += criterion(output_list, y[:, (T-tail):])


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(mylstm.weight_hh_l0.grad)
        epoch_loss += loss.item()

    return epoch_loss / len(dataLoader)

def evaluate(model, device, dataLoader, criterion, tail=5, teacher_force_ratio=0):
    model.to(device)
    model.eval()
    epoch_loss = 0
    for i, batch in enumerate(dataLoader):
        x, y = batch
        N = x.size(0)
        T = x.size(1)
        x = x.to(device)
        y = y.to(device)

        loss = 0
        # use dynamic programming here
        output_list = torch.zeros((N, tail)).double()
        output_list.to(device)

        for j in range(tail):
            xx = x[:, 1:(T - tail + j + 1), ]

            if j == 0:
                yy = y[:, :(T - tail + j)].unsqueeze(2)
            else:
                yy = y[:, :(T - tail + j)].clone().detach()
                # yy =  y[:, :(T-tail+j)].clone().detach().to(device)

                for k in range(j):
                    if np.random.random() > teacher_force_ratio:
                        yy[:, (-j + k)] = output_list[:, k]

                yy = yy.unsqueeze(2)

            xx = torch.cat((xx, yy), 2)
            output = model(xx)
            output_list[:, j] = output[:, -1]


        loss += criterion(output_list, y[:, (T - tail):])


        epoch_loss += loss.item()

    return epoch_loss / len(dataLoader)


def mc_dropout_evaluation(model, device, dataLoader, criterion, tail=5, teacher_force_ratio=1, n_sim=10):
    model.to(device)
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(dataLoader):
        x, y = batch
        N = x.size(0)
        T = x.size(1)

        x = x.to(device)
        y = y.to(device)

        n_output_list = torch.zeros((n_sim, N, tail))
        for n in range(n_sim):
            loss = 0
            # use dynamic programming here
            output_list = torch.zeros((N, tail)).double()
            output_list.to(device)

            for j in range(tail):
                xx = x[:, 1:(T - tail + j + 1), ]

                if j == 0:
                    yy = y[:, :(T - tail + j)].unsqueeze(2)
                else:
                    yy = y[:, :(T - tail + j)].clone().detach()
                    # yy =  y[:, :(T-tail+j)].clone().detach().to(device)

                    for k in range(j):
                        if np.random.random() > teacher_force_ratio:
                            yy[:, (-j + k)] = output_list[:, k]

                    yy = yy.unsqueeze(2)

                xx = torch.cat((xx, yy), 2)
                output = model(xx)
                output_list[:, j] = output[:, -1]

            n_output_list[n, :, :] = output_list


        output_list = torch.mean(n_output_list, dim=0)

        loss += criterion(output_list, y[:, (T-tail):])
        epoch_loss += loss.item()



    return epoch_loss / len(dataLoader)


def mc_dropout_forward_pass(model, device, x, y, tail=5, teacher_force_ratio=1, n_sim=100):

    N = x.size(0)
    T = x.size(1)



    n_output_list = torch.zeros((n_sim, N, tail))
    for n in range(n_sim):
        loss = 0
        # use dynamic programming here
        output_list = torch.zeros((N, tail)).double()
        output_list.to(device)

        for j in range(tail):
            xx = x[:, 1:(T - tail + j + 1), ]

            if j == 0:
                yy = y[:, :(T - tail + j)].unsqueeze(2)
            else:
                yy = y[:, :(T - tail + j)].clone().detach()
                # yy =  y[:, :(T-tail+j)].clone().detach().to(device)

                for k in range(j):
                    if np.random.random() > teacher_force_ratio:
                        yy[:, (-j + k)] = output_list[:, k]

                yy = yy.unsqueeze(2)

            xx = torch.cat((xx, yy), 2)
            output = model(xx)
            output_list[:, j] = output[:, -1]

        n_output_list[n, :, :] = output_list



    mean = torch.mean(n_output_list, dim=0)
    lower = torch.quantile(n_output_list, q=0.05, dim=0)
    upper = torch.quantile(n_output_list, q=0.95, dim=0)


    return mean, lower, upper


def visualization(mean, lower, upper, y_true):
    "input mean, lower, and upper is of size (T,)"

    mean = mean.squeeze().detach().numpy()
    lower = lower.squeeze().detach().numpy()
    upper = upper.squeeze().detach().numpy()
    y_true = y_true.squeeze().detach().numpy()

    T = np.arange(mean.shape[0]) + 1
    fig = go.Figure()
    line_mean = go.Scatter(name='prediction', x=T, y=mean, mode='lines', line=dict(color='rgb(31, 119, 180)'))
    y_true = go.Scatter(name='true', x=T, y=y_true, mode='lines', line=dict(color='rgb(119, 31, 180)'))
    line_upper = go.Scatter(name='Upper Bound',
                            x=T,
                            y=upper,
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            mode='lines',
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            showlegend=False)
    line_lower = go.Scatter(name='Lower Bound',
                            x=T,
                            y=lower,
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            mode='lines',
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            fill='tonexty',
                            showlegend=False)


    fig.add_trace(line_mean)
    fig.add_trace(y_true)
    fig.add_trace(line_upper)
    fig.add_trace(line_lower)

    fig.show()


def simulation(N, T, feature_dim):
    "generate y(t) = x1(t) + x2(t-1) + epsilon(t), where epsilon(t) is auto-regressive"
    # variance of the error

    cur_seq_len = T

    sigma2 = 1

    # begin generating the data
    X = np.empty((N, cur_seq_len, feature_dim))
    Y = np.empty((N, cur_seq_len))

    for i in range(N):
        # generate a random number as the current seqence length
        y = np.empty(cur_seq_len)

        x1 = np.empty(cur_seq_len)
        x2 = np.empty(cur_seq_len)
        x3 = np.ones(cur_seq_len) * np.random.uniform(0,1,1)

        epsilon = np.empty(cur_seq_len)

        # y[1], y[2]
        epsilon[0] = np.random.normal(0, 0.5, 1)
        x1[0] = np.random.uniform(2, 3, 1)
        x2[0] = np.random.uniform(0, 1, 1)
        y[0] = x1[0] + x3[0] + epsilon[0]

        epsilon[1] = epsilon[0] + np.random.normal(0, 0.5, 1)
        x1[1] = np.random.uniform(2, 3, 1)
        x2[1] = np.random.uniform(0, 1, 1)
        y[1] = x1[1] + x2[0] + x3[0] + epsilon[1]

        # after y[2]
        for j in range(2, cur_seq_len):
            epsilon[j] = 0.8 * epsilon[j-1] + 0.2 * epsilon[j-2] + np.random.normal(0, 0.5, 1)
            x1[j] = np.random.uniform(2, 3, 1)
            x2[j] = np.random.uniform(0, 1, 1)
            y[j] = x1[j] + 0.6 * x2[j - 1] + 0.4 * x2[j - 2] + x3[j] + epsilon[j]

        X[i, :, :] = np.transpose(np.array([x1, x2, x3]))
        Y[i, :] = y

    eps = 1e-4
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)

    # minmax scaler
    # max = 1
    # min = -1
    # Y_std = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
    # Y_std = Y_std * (max - min) + min

    Y_std = Y
    return X, Y_std


def simulation_2(N, T, feature_dim):
    "generate y(t) = x1(t) + x2(t-1) + epsilon(t), where epsilon(t) is auto-regressive"
    # variance of the error

    cur_seq_len = T

    sigma2 = 1

    # begin generating the data
    X = np.empty((N, cur_seq_len, feature_dim))
    Y = np.empty((N, cur_seq_len))

    for i in range(N):
        # generate a random number as the current seqence length
        y = np.empty(cur_seq_len)
        x1 = np.empty(cur_seq_len)
        x2 = np.empty(cur_seq_len)


        x1[0] = np.random.uniform(0, 1, 1)
        x2[0] = np.random.uniform(0, 1, 1)
        epsilon = np.random.normal(0, 0.5, 1)
        y[0] = x1[0] + x2[0] + epsilon




        # after y[2]
        for j in range(1, cur_seq_len):
            epsilon = np.random.normal(0, 0.5, 1)
            x1[j] = np.random.uniform(0, 1, 1)
            x2[j] = np.random.uniform(0, 1, 1)
            y[j] = y[j-1] + x1[j] + x2[j] + epsilon

        X[i, :, :] = np.transpose(np.array([x1, x2]))
        Y[i, :] = y

    eps = 1e-4
    X = (X - X.mean()) / (X.std() + eps)

    return X, Y



class DataWrapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx, :, :], self.y[idx, :])








if __name__ == "__main__":

    # x = torch.rand(N, T, FEATURE_DIM)
    # y = torch.rand(N, T)
    # data = DataWrapper(x, y)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', help="choose training mode or test mode")
    opt = parser.parse_args()
    N = 100
    T = 100
    BATCH_SIZE = 1
    FEATURE_DIM = 3
    INPUT_DIM = FEATURE_DIM + 1
    HIDDEN_DIM = 50

    tail = 20
    teacher_force_ratio = 0.5

    X, Y = simulation(N, T, FEATURE_DIM)
    # X, Y = simulation_2(N, T, FEATURE_DIM)
    data = DataWrapper(X, Y)

    training_rate, validation_rate = 0.7, 0.3
    training_size = np.int(N * training_rate)
    validation_size = N - training_size

    training_data, validation_data = torch.utils.data.random_split(data, [training_size, validation_size])
    training_dataLoader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataLoader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
    mylstm = LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, dropouti=0.1,
                  dropoutw=0.1, dropouto=0.1)

    # for i in mylstm.named_parameters():
    #     print(i)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.mode == 'train':

        mylstm.double()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mylstm.parameters(), lr=1e-3)

        n_epochs = 2
        for i in range(n_epochs):
            training_loss = train(mylstm, device, training_dataLoader, optimizer, criterion, tail=tail, teacher_force_ratio=teacher_force_ratio)
            validation_loss = evaluate(mylstm, device, validation_dataLoader, criterion, tail=tail, teacher_force_ratio=0)
            mc_dropout_loss = mc_dropout_evaluation(mylstm, device, validation_dataLoader, criterion, tail=tail, teacher_force_ratio=0, n_sim=10)
            print("---Epoch %d---"%i)
            print("training loss is %.4f"%training_loss)
            print("mc dropout loss is %.4f"%mc_dropout_loss)
            print("validation loss is %.4f"%validation_loss)

        #visualization
        # for i, batch in enumerate(training_dataLoader):
        #     if i == 1:
        #         break
        #
        #     test_point_x, test_point_y = batch
        #     pred_y, lower, upper = mc_dropout_forward_pass(mylstm, device, test_point_x, test_point_y, tail=tail, teacher_force_ratio=0, n_sim=10)
        #
        #     y_true = test_point_y[:, (T-tail):]
        #     visualization(pred_y, lower, upper, y_true)

        torch.save(mylstm.state_dict(), 'model.pt')


    else:


        mylstm.load_state_dict(torch.load('model.pt', map_location=device))
        mylstm.double()

        # visualization
        for i, batch in enumerate(validation_dataLoader):
            if i == 1:
                break

            test_point_x, test_point_y = batch
            pred_y, lower, upper = mc_dropout_forward_pass(mylstm, device, test_point_x, test_point_y, tail=tail, teacher_force_ratio=0, n_sim=10)

            y_true = test_point_y[:, (T - tail):]
            visualization(pred_y, lower, upper, y_true)







