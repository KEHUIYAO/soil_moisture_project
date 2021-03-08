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
    def __init__(self, csv_file,
                 time_varying_features_name = ['prcp', 'srad', 'tmax', 'tmin', 'vp','SMAP_36km'],
                 static_features_name = ['elevation', 'slope', 'aspect', 'hillshade', 'clay', 'sand', 'bd', 'soc', 'LC'],
                 transform = None):
        data = pd.read_csv(csv_file)
        data['formatted_date'] = pd.to_datetime(data.Date, format='%Y%m%d')
        # create a column called mask, which indicates whether SMAP_1km data is missing or not
        data['mask'] = 1
        data['mask'][pd.isna(data.SMAP_1km)] = 0
        data['SMAP_36km'] = data['SMAP_36km'].fillna(method = 'ffill')
        data['SMAP_36km'] = data['SMAP_36km'].fillna(method = 'bfill')

        data['month'] = data['formatted_date'].dt.month
        data['year'] = data['formatted_date'].dt.year
        data['SMAP_1km'][pd.isna(data['SMAP_1km'])] = 0
        data['index'] = list(range(data.shape[0]))

        # time varying features
        features = data[time_varying_features_name]
        # normalize the features
        features = (features - features.mean()) / (features.std()+1e-5)

        # time independent features
        static = data[static_features_name]
        # normalize the features
        static = (static - static.mean()) / (static.std()+1e-5)

        self.data = data
        self.features = features
        self.static = static

        # use one year's data for training
        self.ind_list = data['index'][~data.duplicated(subset=['POINTID', 'year'])]

        self.ind_list = list(self.ind_list) + [data.shape[0]]
        self.transform = transform


    def __len__(self):
        # return number of 1-year-seqential data in the dataset
        return len(self.ind_list) - 1

    def __getitem__(self,idx):
        """

        :param idx: get the idx sample from the dataset
        :return: x:(seq_len,1), mask:(seq_len,1), features:(seq_len,5), static:(seq_len,9)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mask = self.data['mask'].iloc[self.ind_list[idx]:self.ind_list[idx+1]].values
        x = self.data['SMAP_1km'].iloc[self.ind_list[idx]:self.ind_list[idx+1]].values

        time_varying_features = self.features.iloc[self.ind_list[idx]:self.ind_list[idx+1],:].values
        static_features = self.static.iloc[self.ind_list[idx]:self.ind_list[idx+1],:].values

        # let the seqence start with a non-missing observation
        ind = 0
        while ind < len(mask):
            if mask[ind] == 1:
                break
            else:
                ind += 1

        # reformat the data
        x = x[ind:].reshape([-1,1])
        mask = mask[ind:].reshape([-1,1])
        features = time_varying_features[ind:,:].reshape([-1,time_varying_features.shape[1]])
        static = static_features[ind:,:].reshape([-1,static_features.shape[1]])
        sample = (x, mask, features,static)

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    mydata = SoilMoistureDataset("../../SMAP_Climate_In_Situ.csv")
    (x, mask, time_varying_features, static_features) = mydata[0]
    print(x.shape)
    print(mask.shape)
    print(time_varying_features.shape)
    print(static_features.shape)
    print("x is", x)
    print("mask is", mask)
    print("time varying features are", time_varying_features)
    print("static features are", static_features)
    print(len(mydata))
