# Load built-in packages
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

# load my functions
from transformer import *
from load_data import SoilMoistureDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/Users/kehuiyao/Desktop/soil moisture/src')


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(model):
    "assemble the varying-rate optimizer"
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def str2bool(v):
    "convert str to bool"
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Rsquare(x, y):
    "calculate the rsquare between two vectors"
    numerator = torch.sum((x - x.mean()) * (y - y.mean()))**2
    denominator = torch.sum((y - y.mean())**2) * torch.sum((x - x.mean())**2)
    return  numerator / denominator

def Rsquare_loss(x, y):
    "use rsquare as a loss function"
    return - Rsquare(x,y)

def count_parameters(model):
    "count how many parameter in the model"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    "inititialization of the weights using xavier initialization"
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

def epoch_time(start_time, end_time):
    "record the time elapsed during one epoch"
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def add_training_mask(mask, p=0.7):
    "randomly mask some true data and ask our model to predict them"
    device = mask.device
    size = mask.size(-2)
    prob = torch.ones([1, size, 1]) * p
    # mask for the transformer
    input_mask = ((torch.bernoulli(prob) == 1).to(device) & (mask == 1)).type_as(mask.data)
    training_mask = ((input_mask == 0) & (mask == 1)).type_as(mask.data)
    return training_mask, input_mask

def train(model, device, dataLoader, optimizer, criterion, clip, use_training_mask):
    """
    when the use_training_mask is True, we randomly mask out some true labels and ask our model to predict these, in this
    situation, we can use both previous and future observations of y.

    when the use_training_mask is False, we do not allow the current prediction depends on the future because each point
    is used in calculating the loss. For example, if we allow position 1 to see position 4. It remembers the value in
    position 4, later when we want to predict position 4 , we can directly copy information from position 1's memory, by which
    makes the training loss converges to 0.
    """
    model.to(device)
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataLoader):
        x, mask, features, static = batch
        # decide to use which training scheme
        if use_training_mask:
            training_mask, input_mask = add_training_mask(mask)
        else:
            training_mask, input_mask = mask, mask

        x = x.to(device)
        input_mask = input_mask.to(device)
        training_mask = training_mask.to(device)
        features = features.to(device)
        static = static.to(device)

        # combine time varying features and static features together
        features = torch.cat([features, static], dim=2)

        src_mask = None
        # if use_training_mask, we allow seeing the future, otherwise, we do not allow seeing the future
        trt_mask = make_std_mask(input_mask, see_future = use_training_mask)

        # clear out the gradients of Variables
        optimizer.zero_grad()
        output = model(features, x, src_mask, trt_mask)

        # when computing the loss function, use training mask
        training_mask = training_mask == 1
        loss = criterion(output[:,:-1,:][training_mask[:,1:,:]], x[:,1:,:][training_mask[:,1:,:]])
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataLoader)


def evaluate(model,device, dataLoader, criterion, use_training_mask):
    "evaluate the model's performance on the validation set"
    model.to(device)
    model.eval()
    epoch_loss = 0
    rsquare_total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataLoader):
            x, mask, features, static = batch
            x = x.to(device)
            if use_training_mask:
                training_mask, input_mask = add_training_mask(mask, 0.5)
            else:
                training_mask, input_mask = mask, mask

            training_mask = training_mask.to(device)
            input_mask = input_mask.to(device)
            features = features.to(device)
            static = static.to(device)

            # combine time varying features and static features together
            features = torch.cat([features, static], dim=2)

            src_mask = None
            tgt_mask = make_std_mask(input_mask, use_training_mask)
            output = model(features, x, src_mask, tgt_mask)

            training_mask = training_mask == 1
            loss = criterion(output[:,:-1,:][training_mask[:,1:,:]], x[:,1:,:][training_mask[:,1:,:]])
            # Rsquare
            rsquare = Rsquare(output[0,:-1,0][training_mask[0,1:,0]], x[0,1:,0][training_mask[0,1:,0]])

            # cumulate the loss
            epoch_loss += loss.item()
            rsquare_total += rsquare.item()

    return epoch_loss / len(dataLoader), rsquare_total / len(dataLoader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default=20, help = "number of epochs to train")
    parser.add_argument("--save_model", type = str, default='model.pt', help = "file to save the model")
    parser.add_argument("--save_figure", type = str, default='model.png', help = "file to save the training and validation performance through epochs")
    parser.add_argument("--save_entire_model", type = str, default='model_entire.pt', help = "file to save the entire model")
    parser.add_argument("--load_model", type = str, default=None, help = 'if specified model name, load pre-trained model')
    parser.add_argument("--load_data", type = str, default="SMAP_Climate_In_Situ_Kenaston_training_data.csv", help = 'file name of the dataset')
    parser.add_argument("--num_layers", type = int, default=2)
    parser.add_argument("--time_varying_features_name", type = str, default='prcp,srad,tmax,tmin,vp,SMAP_36km', help = "name of time varying features included")
    parser.add_argument("--static_features_name", type = str, default='elevation,slope,aspect,hillshade,clay,sand,bd,soc,LC', help = 'name of static features included')
    parser.add_argument("--dropout", type = float, default= 0.1, help = 'the dropout rate in lstm')
    parser.add_argument("--early_stopping_patience", type = int, default = 10, help = 'set the maximum number of epochs we should wait when the validation error does not decrease')
    parser.add_argument("--criterion", type = str, default='mse', help='choose mse or rsquare as the loss')
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--varying_learning_rate", type=str, default='true')
    parser.add_argument("--use_training_mask", type=str, default='true')
    opt = parser.parse_args()

    # set random seed to 0
    #np.random.seed(0)
    #torch.manual_seed(0)

    # load the data and making training set
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
    d_feature = len(static_features_name) + len(time_varying_features_name)
    N = opt.num_layers
    d_model = opt.d_model
    d_ff = opt.d_ff
    h = opt.h
    dropout = opt.dropout
    model = make_model(d_feature, N, d_model, d_ff, h, dropout)
    model.double()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("I am running on the", device)

    # loss criteria
    if opt.criterion == 'mse':
        criterion = nn.MSELoss()
    elif opt.criterion == 'rsquare':
        criterion = Rsquare_loss

    # optimizer
    if str2bool(opt.varying_learning_rate):
        optimizer = get_std_opt(model)
    else:
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
        train_loss = train(model, device, training_dataLoader, optimizer, criterion, clip, use_training_mask)
        valid_loss, rsquare = evaluate(model, device, validation_dataLoader, criterion, use_training_mask)

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
        print(f'\tTrain Loss: {train_loss:.8f} | Train PPL: {math.exp(train_loss):.8f}')
        print(f'\t Val. Loss: {valid_loss:.8f} |  Val. PPL: {math.exp(valid_loss):.8f}')
        print("rsquare is %f"%rsquare)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train_loss_list))+1,np.array(train_loss_list), label = "training")
    ax.plot(np.arange(len(valid_loss_list))+1,np.array(valid_loss_list), label = "validation")
    ax.legend()
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.savefig(opt.save_figure)


