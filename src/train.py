"""### Load pytorch packages."""

import torch.nn as nn
import torch
import torch.optim as optim
import time
import math
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/Users/kehuiyao/Desktop/soil moisture/src')


from model import SoilMoistureGapFilling
from load_data import SoilMoistureDataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Rsquare(x, y):
    """

    :param x: x is a one dimensional vector
    :param y: y is a one dimensional vector
    :return: a scalar between (0,1)
    """
    numerator = torch.sum((x - x.mean()) * (y - y.mean())) ** 2
    denominator = torch.sum((y - y.mean()) ** 2) * torch.sum((x - x.mean()) ** 2)

    return numerator / denominator


def Rsquare_loss(x, y):
    """

    :param x: x is a one dimensional vector
    :param y: y is a one dimensional vector
    :return: a scalar between (-1,0)
    """

    return - Rsquare(x, y)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def add_training_mask(mask, p=0.7 ):
    "randomly mask some true data and ask our model to predict them"
    device = mask.device
    size = mask.size(-2)
    prob = torch.ones([1, size, 1]) * p
    # mask for the transformer
    input_mask = ((torch.bernoulli(prob) == 1).to(device) & (mask == 1)).type_as(mask.data)

    # the first element should not be masked in training
    input_mask[0,0,0] = 1

    training_mask = ((input_mask == 0) & (mask == 1)).type_as(mask.data)
    return training_mask, input_mask

def train(model, device, dataLoader, optimizer, criterion, clip, teacher_force_ratio=1, use_training_mask=False):
    model.to(device)
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataLoader):
        x, mask, features, static = batch
        x = x.to(device)
        mask = mask.to(device)

        # add masking
        if use_training_mask:
            training_mask, input_mask = add_training_mask(mask, 0.5)
        else:
            training_mask, input_mask = mask, mask

        features = features.to(device)
        static = static.to(device)
        optimizer.zero_grad()
        output = model(x, input_mask, features, static, teacher_force_ratio)
        # print(output.size())
        # print(mask.size())
        # print(x.size())

        training_mask = training_mask > 0
        loss = criterion(output[:, :-1, :][training_mask[:, 1:, :]], x[:, 1:, :][training_mask[:, 1:, :]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataLoader)


def evaluate(model, device, dataLoader, criterion, teacher_force_ratio, use_training_mask=False):
    model.to(device)
    model.eval()
    epoch_loss = 0
    rsquare_total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataLoader):
            x, mask, features, static = batch
            x = x.to(device)
            mask = mask.to(device)

            # add masking
            if use_training_mask:
                training_mask, input_mask = add_training_mask(mask, 0.5)
            else:
                training_mask, input_mask = mask, mask


            features = features.to(device)
            static = static.to(device)

            # teacher force evaluation, which micmics the situation of temporal gap filling
            output = model(x, input_mask, features, static, teacher_force_ratio)

            training_mask = training_mask > 0

            loss = criterion(output[:, :-1, :][training_mask[:, 1:, :]], x[:, 1:, :][training_mask[:, 1:, :]])

            # Rsquare
            rsquare = Rsquare(output[0, :-1, 0][training_mask[0, 1:, 0]], x[0, 1:, 0][training_mask[0, 1:, 0]])

            # cumulate the loss
            epoch_loss += loss.item()
            rsquare_total += rsquare.item()

    return epoch_loss / len(dataLoader), rsquare_total / len(dataLoader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--ratio", type=float, default=1, help="the teacher force ratio ranging from 0 to 1")
    parser.add_argument("--save_model", type=str, default='model.pt', help="file to save the model")
    parser.add_argument("--save_figure", type=str, default='model.png',
                        help="file to save the training and validation performance through epochs")
    parser.add_argument("--save_entire_model", type=str, default='model_entire.pt',
                        help="file to save the entire model")

    parser.add_argument("--load_model", type=str, default=None, help='if specified model name, load pre-trained model')
    parser.add_argument("--parallel", type=str, default='false', help='if True, enable parallel training')
    parser.add_argument("--direct_connection_from_previous_output", type=str, default='true',
                        help='if True, add direct connection from previous output to current input')
    parser.add_argument("--bias", type=str, default='true', help='if True, include bias term in the lstm')
    parser.add_argument("--num_layers", type=int, default=1, help='number of layers used in the lstm network')
    # parser.add_argument("--bidirectional", type = str, default='false', help = 'if True, use bidirectional lstm')
    parser.add_argument("--load_data", type=str, default="../../simulation_training_data.csv",
                        help='file name of the dataset')

    parser.add_argument("--time_varying_features_name", type=str, default='x1,x2',
                        help="name of time varying features included")
    parser.add_argument("--static_features_name", type=str, default='x3', help='name of static features included')
    parser.add_argument("--lstm_hidden_dim", type=int, default=128, help='the hidden dim of the lstm')
    parser.add_argument("--ffn_hidden_dim", type=int, default=128, help='the hidden dim of the ffn')
    parser.add_argument("--dropout", type=float, default=0, help='the dropout rate in lstm')
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help='set the maximum number of epochs we should wait when the validation error does not decrease')
    parser.add_argument("--criterion", type=str, default='mse', help='choose mse or rsquare as the loss')
    parser.add_argument("--use_training_mask", type=str, default='false',
                        help='choose whether using training mask or not')

    opt = parser.parse_args()

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load the data and making training set
    # """### Use DataLoader to store the data"""

    time_varying_features_name = opt.time_varying_features_name.split(',')

    static_features_name = opt.static_features_name.split(',')

    data = SoilMoistureDataset(opt.load_data, time_varying_features_name, static_features_name)

    BATCH_SIZE = 1
    N = len(data)
    training_rate, validation_rate = 0.7, 0.3

    training_size = np.int(N * training_rate)
    validation_size = N - training_size

    training_data, validation_data = torch.utils.data.random_split(data, [training_size, validation_size])
    training_dataLoader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataLoader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

    # build the model

    # model

    time_varying_dim = len(time_varying_features_name)
    static_dim = len(static_features_name)

    lstm_hidden_dim = opt.lstm_hidden_dim
    ffn_hidden_dim = opt.ffn_hidden_dim
    num_layers = opt.num_layers
    # bidirectional = str2bool(opt.bidirectional)
    bidirectional = False
    bias = str2bool(opt.bias)
    dropout = opt.dropout
    direct_connection_from_previous_output = str2bool(opt.direct_connection_from_previous_output)
    # print(opt.direct_connection_from_previous_output)
    print(direct_connection_from_previous_output)

    model = SoilMoistureGapFilling(time_varying_dim, static_dim, lstm_hidden_dim, ffn_hidden_dim, num_layers,
                                   bidirectional, bias, dropout, direct_connection_from_previous_output)
    model.double()

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("I am running on the", device)

    if torch.cuda.device_count() > 1 and str2bool(opt.parallel):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    # loss criteria
    if opt.criterion == 'mse':
        criterion = nn.MSELoss()
    elif opt.criterion == 'rsquare':
        criterion = Rsquare_loss

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # begin to train
    # see how many parameters in the model
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # load from a pre-trained model if possible
    if opt.load_model:
        model.load_state_dict(torch.load(opt.load, map_location=device))
    else:
        # initialize the model
        model.apply(init_weights)

    # number of epochs to train
    n_epochs = opt.epochs
    # set the teacher force ratio
    teacher_force_ratio = opt.ratio
    # gradient clipping
    clip = 1

    # training and validation result for each epoch
    train_loss_list = []
    valid_loss_list = []
    early_stopping_patience = opt.early_stopping_patience
    best_loss = float('inf')
    count = 0
    for epoch in range(n_epochs):

        if count > early_stopping_patience:
            break

        start_time = time.time()

        use_training_mask = str2bool(opt.use_training_mask)

        train_loss = train(model, device, training_dataLoader, optimizer, criterion, clip, teacher_force_ratio,
                           use_training_mask)
        valid_loss, rsquare = evaluate(model, device, validation_dataLoader, criterion, teacher_force_ratio,
                                       use_training_mask)

        # early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            count = 0
            # save state dict
            torch.save(model.state_dict(), opt.save_model)
            # save the entire model
            torch.save(model, opt.save_entire_model)

        else:
            count = count + 1

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.8f} | Train PPL: {math.exp(train_loss):.8f}')
        # print(f'\t Val. Loss: {valid_loss:.8f} |  Val. PPL: {math.exp(valid_loss):.8f}')
        print(train_loss)
        print(valid_loss)
        print("rsquare is %f" % rsquare)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train_loss_list)) + 1, np.array(train_loss_list), label="training")
    ax.plot(np.arange(len(valid_loss_list)) + 1, np.array(valid_loss_list), label="validation")
    ax.legend()
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.savefig(opt.save_figure)
