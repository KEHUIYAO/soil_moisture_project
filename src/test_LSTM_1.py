
import torch.nn as nn
import torch
import torch.optim as optim
import time
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/kehuiyao/Desktop/soil moisture/src')


from LSTM_1 import LSTM_1
from load_data import SoilMoistureDataset
from utility import running_mean


# """### Use DataLoader to store the data"""
data = SoilMoistureDataset("../data/SMAP_Climate_In_Situ.csv")
BATCH_SIZE = 1
N = len(data)
training_rate, validation_rate, test_rate = 0.6, 0.3, 0.1
#training_size, validation_size = np.int(N * training_rate), np.int(N * validation_rate)
training_size, validation_size = 1000,3
test_size = N - training_size - validation_size

training_data, validation_data, testing_data = torch.utils.data.random_split(data,[training_size, validation_size, test_size])
training_dataLoader = torch.utils.data.DataLoader(training_data,batch_size = BATCH_SIZE,shuffle = False)
validation_dataLoader = torch.utils.data.DataLoader(validation_data,batch_size= BATCH_SIZE,shuffle = False)
testing_dataLoader = torch.utils.data.DataLoader(testing_data,batch_size = BATCH_SIZE,shuffle = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 10

NUM_LAYERS = 1
BIAS = True

model = LSTM_1(HIDDEN_DIM, NUM_LAYERS, BIAS)
model.double()

model.load_state_dict(torch.load('LSTM_1.pt'))
model.eval()
criterion = nn.MSELoss()

def evaluate(model, dataLoader, criterion):
    model.to(device)
    model.eval()
    epoch_loss = 0
    # one step backward
    one_step_backward_total_loss = 0
    running_mean_backward_total_loss = 0
    god_mean_total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataLoader):
            x, mask = batch
            x = x[0, :]
            mask = mask[0, :]
            x = x.to(device)
            mask = mask.to(device)

            output = model(x, mask)

            mask = mask > 0

            loss = criterion(output[:-1][mask[1:]], x[1:][mask[1:]])

            # one step backward prediction
            one_step_backward_pred = x[mask][:-1]
            one_step_backward_true = x[mask][1:]
            one_step_backward_loss = criterion(one_step_backward_pred, one_step_backward_true)


            # running mean backward prediction
            running_mean_backward_pred = running_mean(x[mask],2)[:-1]
            running_mean_backward_true = x[mask][2:]
            running_mean_backward_loss = criterion(running_mean_backward_pred, running_mean_backward_true)


            # God prediction baseline
            god_pred = torch.ones_like(x[mask])*torch.mean(x[mask])
            god_true = x[mask]
            god_loss = criterion(god_pred, god_true)



            print(x)
            print(x[1:][mask[1:]])
            print(output)
            print(output[:-1][mask[1:]])



            epoch_loss += loss.item()
            one_step_backward_total_loss += one_step_backward_loss.item()
            running_mean_backward_total_loss += running_mean_backward_loss.item()
            god_mean_total_loss += god_loss.item()



    return epoch_loss / len(dataLoader), one_step_backward_total_loss / len(dataLoader), running_mean_backward_total_loss / len(dataLoader), god_mean_total_loss / len(dataLoader)




valid_loss, one_step_backward_loss, running_mean_backward_loss, god_loss = evaluate(model, validation_dataLoader, criterion)



print(f'\t Val. Loss: {valid_loss:.5f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
print(f'\t One step backward: {one_step_backward_loss:.5f} |  Val. PPL: {math.exp(one_step_backward_loss):7.3f}')
print(f'\t Running mean backward: {running_mean_backward_loss:.5f} |  Val. PPL: {math.exp(running_mean_backward_loss):7.3f}')
print(f'\t God mean backward: {god_loss:.5f} |  Val. PPL: {math.exp(god_loss):7.3f}')
