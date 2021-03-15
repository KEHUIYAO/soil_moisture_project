from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class SoilMoistureDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        """

        :param csv_file: Path to the csv file
        :param transform: Optional transform to be applied on a sample
        """


        data = pd.read_csv(csv_file)
        # do not use Insitu observations
        data = data.iloc[:946080, ]
        data = data[~pd.isna(data.SMAP_1km)]


        # time varying features
        features = data[['prcp', 'srad', 'tmax', 'tmin', 'vp','elevation', 'slope', 'aspect', 'hillshade', 'clay', 'sand', 'bd', 'soc', 'LC']]
        # normalize the features
        features = (features - features.mean()) / features.std()





        self.data = data
        self.features = features







        self.transform = transform


    def __len__(self):
        # return number of 1-year-seqential data in the dataset
        return len(self.data)

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.data['SMAP_1km'].iloc[idx].reshape(1)
        x = self.features.iloc[idx,:].values

        return x, y


if __name__ == "__main__":

    mydata = SoilMoistureDataset("../data/SMAP_Climate_In_Situ.csv")
    (x, y) = mydata[0]
    print(x)
    print(y)
