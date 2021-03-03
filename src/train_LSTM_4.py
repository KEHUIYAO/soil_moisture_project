

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
sys.path.insert(1, '/Users/kehuiyao/Desktop/soil moisture/src')


from LSTM_4 import LSTM_4
from load_data_3 import SoilMoistureDataset
from utility import running_mean






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




def train(model, device, dataLoader, optimizer, criterion, clip, teacher_force_ratio = 1):
    model.to(device)
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataLoader):
        x, mask, features, static = batch
        x = x.to(device)
        mask = mask.to(device)
        features = features.to(device)
        static = static.to(device)
        optimizer.zero_grad()
        output = model(x, mask, features, static, teacher_force_ratio)
        # print(output.size())
        # print(mask.size())
        # print(x.size())


        mask = mask > 0

        loss = criterion(output[:,:-1,:][mask[:,1:,:]], x[:,1:,:][mask[:,1:,:]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataLoader)


def evaluate(model,device, dataLoader, criterion, teacher_force_ratio):
    model.to(device)
    model.eval()
    epoch_loss = 0
    rsquare_total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataLoader):
            x, mask, features, static = batch
            x = x.to(device)
            mask = mask.to(device)
            features = features.to(device)
            static = static.to(device)

            # teacher force evaluation, which micmics the situation of temporal gap filling
            output = model(x, mask, features, static, teacher_force_ratio)



            mask = mask > 0

            loss = criterion(output[:,:-1,:][mask[:,1:,:]], x[:,1:,:][mask[:,1:,:]])



            # Rsquare
            rsquare = np.corrcoef(output[0,:-1,0][mask[0,1:,0]].detach().cpu().numpy(), x[0,1:,0][mask[0,1:,0]].detach().cpu().numpy())[0,1]**2


            # cumulate the loss
            epoch_loss += loss.item()
            rsquare_total += rsquare


    return epoch_loss / len(dataLoader), rsquare_total / len(dataLoader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default=20, help = "number of epochs to train")
    parser.add_argument("--ratio", type = float, default=1, help = "the teacher force ratio")
    parser.add_argument("--save", type = str, default='model', help = "file to save the plots and model")
    parser.add_argument("--mode", type = str, default='train', help = "set training mode or test mode")
    parser.add_argument("--load", type = str, default='model.pt', help = 'load pre-trained model')

    opt = parser.parse_args()

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load the data and making training set
    # """### Use DataLoader to store the data"""
    data = SoilMoistureDataset("../data/SMAP_Climate_In_Situ.csv")

    BATCH_SIZE = 1
    N = len(data)
    training_rate, validation_rate, test_rate = 0.6, 0.3, 0.1
    training_size, validation_size = np.int(N * training_rate), np.int(N * validation_rate)
    #training_size, validation_size = 100,10
    test_size = N - training_size - validation_size
    training_data, validation_data, testing_data = torch.utils.data.random_split(data, [training_size, validation_size,
                                                                         test_size])
    training_dataLoader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataLoader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
    testing_dataLoader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

    # build the model
    FEATURE_DIM = 5
    STATIC_DIM = 9
    HIDDEN_DIM_LSTM = 50
    HIDDEN_DIM_FFN = 20

    model = LSTM_4(FEATURE_DIM, STATIC_DIM, HIDDEN_DIM_LSTM, HIDDEN_DIM_FFN)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    if opt.mode == "train":

        # begin to train
        # see how many parameters in the model
        print(f'The model has {count_parameters(model):,} trainable parameters')


        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = train(model, device, training_dataLoader, optimizer, criterion, clip, teacher_force_ratio)
            valid_loss, rsquare = evaluate(model, device, validation_dataLoader, criterion, teacher_force_ratio)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)


            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.8f} | Train PPL: {math.exp(train_loss):.8f}')
            print(f'\t Val. Loss: {valid_loss:.8f} |  Val. PPL: {math.exp(valid_loss):.8f}')
            print("rsquare is %f"%rsquare)

        fig, ax = plt.subplots()
        ax.plot(np.arange(n_epochs)+1,np.array(train_loss_list), label = "training")
        ax.plot(np.arange(n_epochs)+1,np.array(valid_loss_list), label = "validation")
        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        fig.savefig(opt.save+".png")


        torch.save(model.state_dict(), opt.save+".pt")

    elif opt.mode == "test":
        model.load_state_dict(torch.load(opt.load))
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        teacher_force_ratio = opt.ratio

        valid_loss, rsquare = evaluate(model, device, testing_dataLoader, criterion, teacher_force_ratio)


        print(f'\t Val. Loss: {valid_loss:.5f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print("rsquare is %f"%rsquare)


