from LSTM_3 import LSTM_3
from LSTM_4 import LSTM_4
from load_data_3 import SoilMoistureDataset
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

def evaluate_plus(model, device, dataLoader, criterion, teacher_force_ratio):
    model.to(device)
    model.eval()
    epoch_loss = 0
    rsquare_total = 0
    dataset = dataLoader.dataset

    # store {rsquare: sample id}
    rsquare_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            x, mask, features, static = batch

            # convert to tensor
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
            features = torch.from_numpy(features)
            features = features.unsqueeze(0)
            static = torch.from_numpy(static)
            static = static.unsqueeze(0)

            # teacher force evaluation, which micmics the situation of temporal gap filling
            output = model(x, mask, features, static, teacher_force_ratio)

            mask = mask > 0

            loss = criterion(output[:, :-1, :][mask[:, 1:, :]], x[:, 1:, :][mask[:, 1:, :]])

            # Rsquare
            rsquare = np.corrcoef(output[0, :-1, 0][mask[0, 1:, 0]].detach().cpu().numpy(),
                                  x[0, 1:, 0][mask[0, 1:, 0]].detach().cpu().numpy())[0, 1] ** 2

            #print(rsquare)

            # cumulate the loss
            epoch_loss += loss.item()
            rsquare_total += rsquare
            rsquare_list.append(rsquare)

    # plot 25th, 50th, 75th, 90th quantile of predicted SMAP_1km corresponding to the rsquare





    for j,i in enumerate([25,50,75,90]):
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()

        pcen = np.percentile(rsquare_list, i, interpolation='nearest')
        #print(pcen)
        id = abs(rsquare_list - pcen).argmin()


        x, mask, features, static = dataset[id]

        # convert to tensor
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        features = torch.from_numpy(features)
        features = features.unsqueeze(0)
        static = torch.from_numpy(static)
        static = static.unsqueeze(0)

        output = model(x, mask, features, static, teacher_force_ratio)
        predicted = output[:, :-1, :].view(-1).detach().cpu().numpy()
        time = np.arange(len(predicted)) + 1
        ax2.plot(time, predicted, label='predicted')
        ax2.legend()
        ax2.set_title(str(i)+'quantile')

        mask = mask > 0
        output, x = output[:, :-1, :][mask[:, 1:, :]].view(-1).detach().cpu().numpy(), x[:, 1:, :][mask[:, 1:, :]].view(-1).detach().cpu().numpy()

        time = np.arange(len(x))+1
        ax.plot(time, x, label = 'true')
        ax.plot(time, output, label = 'predicted')
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        ax.set_title(str(i)+' quantile')


        plt.show()




    return epoch_loss / len(dataLoader), rsquare_total / len(dataLoader)


if __name__ == "__main__":
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load the data and making training set
    # """### Use DataLoader to store the data"""
    # data = SoilMoistureDataset("../../SMAP_Climate_In_Situ.csv")
    # data = SoilMoistureDataset("../../SMAP_Climate_In_Situ_Kenaston.csv")
    data = SoilMoistureDataset("../../SMAP_Climate_In_Situ_TxSON.csv")
    BATCH_SIZE = 1
    N = len(data)
    training_rate, validation_rate, test_rate = 0.6, 0.3, 0.1
    training_size, validation_size = np.int(N * training_rate), np.int(N * validation_rate)
    # training_size, validation_size = 10,1
    test_size = N - training_size - validation_size
    training_data, validation_data, testing_data = torch.utils.data.random_split(data, [training_size, validation_size,
                                                                                        test_size])
    # training_dataLoader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    #validation_dataLoader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
    testing_dataLoader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True)

    # build the model for LSTM_3
    FEATURE_DIM = 5
    STATIC_DIM = 9
    HIDDEN_DIM_LSTM = 128
    HIDDEN_DIM_FFN = 128

    # build the model for LSTM_4
    # FEATURE_DIM = 5
    # STATIC_DIM = 9
    # HIDDEN_DIM_LSTM = 128
    # HIDDEN_DIM_FFN = 128
    # bidirectional = False



    # model for LSTM_3
    model = LSTM_3(FEATURE_DIM, STATIC_DIM, HIDDEN_DIM_LSTM, HIDDEN_DIM_FFN)
    model.double()
    model_file = 'LSTM_3_with_teacher.pt'
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))

    # model for LSTM_4
    # model = LSTM_4(FEATURE_DIM, STATIC_DIM, HIDDEN_DIM_LSTM, HIDDEN_DIM_FFN, bidirectional)
    # model.double()
    # model_file = 'LSTM_4_with_teacher.pt'
    # model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
    #
    #



    # loss criteria
    criterion = nn.MSELoss()
    device = torch.device('cpu')
    teacher_force_ratio = 1

    # evaluate the model
    loss, rsquare = evaluate_plus(model,device, testing_dataLoader, criterion, teacher_force_ratio)
    print("loss is %f"%loss)
    print("Average R square is %f"%rsquare)
