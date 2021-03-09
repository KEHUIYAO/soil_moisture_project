from model import SoilMoistureGapFilling
import argparse
from load_data import SoilMoistureDataset
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

def Rsquare(x, y):
    """

    :param x: x is a one dimensional vector
    :param y: y is a one dimensional vector
    :return: a scalar between (0,1)
    """
    numerator = torch.sum((x - x.mean()) * (y - y.mean()))**2
    denominator = torch.sum((y - y.mean())**2) * torch.sum((x - x.mean())**2)

    return  numerator / denominator

def test(model, device, dataLoader, criterion, teacher_force_ratio = 1):
    model.to(device)
    model.eval()
    epoch_loss = 0
    rsquare_total = 0
    dataset = dataLoader.dataset

    # store rsquare of each true predicted pair
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
            rsquare = Rsquare(output[0, :-1, 0][mask[0, 1:, 0]],
                                  x[0, 1:, 0][mask[0, 1:, 0]])


            #print(rsquare)

            # cumulate the loss
            epoch_loss += loss.item()
            rsquare_total += rsquare.item()
            rsquare_list.append(rsquare.item())

    # plot 25th, 50th, 75th, 90th quantile of predicted SMAP_1km corresponding to the rsquare





    for j,i in enumerate([25,50,75,90]):
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()

        pcen = np.percentile(rsquare_list, i, interpolation='nearest')

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", type = str, default="../../SMAP_Climate_In_Situ_Kenaston_testing_data.csv", help = 'file name of the dataset')
    parser.add_argument("--load_entire_model", type = str, default="../conda/model_entire_2.pt", help = 'file name to load the entire model')

    opt = parser.parse_args()

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load the testing dataset
    data = SoilMoistureDataset(opt.load_data)
    BATCH_SIZE = 1
    N = len(data)
    testing_dataLoader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(opt.load_entire_model, map_location=torch.device(device))
    model.eval()




    # loss criteria
    criterion = nn.MSELoss()
    device = torch.device('cpu')


    # evaluate the model
    loss, rsquare = test(model,device, testing_dataLoader, criterion)
    print("loss is %f"%loss)
    print("Average R square is %f"%rsquare)
