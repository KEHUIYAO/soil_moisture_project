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
    def __init__(self, csv_file, transform = None, include_features = None, include_static = None, unit = 'month'):
        """

        :param csv_file: Path to the csv file
        :param transform: Optional transform to be applied on a sample
        """


        data = pd.read_csv(csv_file)
        data = data.iloc[:946080, ]
        data['formatted_date'] = pd.to_datetime(data.Date, format='%Y%m%d')
        data['mask'] = 1
        data['mask'][pd.isna(data.SMAP_36km)] = 0
        data['month'] = data['formatted_date'].dt.month
        data['year'] = data['formatted_date'].dt.year
        data['SMAP_36km'][pd.isna(data['SMAP_36km'])] = 0
        data['index'] = list(range(data.shape[0]))

        # time varying features
        features = data[['prcp', 'srad', 'tmax', 'tmin', 'vp']]
        # normalize the features
        features = (features - features.mean()) / features.std()

        # time independent features
        static = data[['elevation', 'slope', 'aspect', 'hillshade', 'clay', 'sand', 'bd', 'soc', 'LC']]
        # normalize the features
        static = (static - static.mean()) / static.std()




        self.data = data
        self.features = features
        self.static = static


        if unit == "month":
            # use one month's data
            self.ind_list = data['index'][~data.duplicated(subset=['POINTID', 'month', 'year'])]
        elif unit == "year":
            # use one year's data
            self.ind_list = data['index'][~data.duplicated(subset=['POINTID', 'year'])]
        else:
            raise ValueError("Time unit not valid")

        self.ind_list = list(self.ind_list) + [data.shape[0]]
        self.transform = transform
        self.include_features = include_features
        self.include_static = include_static

    def __len__(self):
        return len(self.ind_list) - 1

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mask = self.data['mask'].iloc[self.ind_list[idx]:self.ind_list[idx+1]].values
        x = self.data['SMAP_36km'].iloc[self.ind_list[idx]:self.ind_list[idx+1]].values
        start = 0
        ind = 0
        while ind < len(mask):
            if mask[ind] == 1:
                break
            else:
                ind += 1



        x = x[ind:]
        mask = mask[ind:]

        if self.include_features and self.include_static:
            features = self.features.iloc[self.ind_list[idx]:self.ind_list[idx+1],:].values
            static = self.static.iloc[self.ind_list[idx]:self.ind_list[idx+1],:].values
            features = features[ind:,:]
            static = static[ind:,:]
            sample = (x, mask, features,static)
        elif self.include_features:
            features = self.features.iloc[self.ind_list[idx]:self.ind_list[idx+1],:].values
            features = features[ind:,:]
            sample = (x, mask, features)
        elif self.include_static:
            static = self.static.iloc[self.ind_list[idx]:self.ind_list[idx+1],:].values
            static = static[ind:,:]
            sample = (x, mask, static)
        else:
            sample = (x, mask)



        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    mydata = SoilMoistureDataset("../data/SMAP_Climate_In_Situ.csv", None, True, True, 'year')
    (x, mask, features, static) = mydata[0]
    print(x.shape)
    print(mask.shape)
    print(features.shape)
    print(static.shape)
    print("x is", x)
    print("mask is", mask)
    print("features are", features)
    print("static features are", static)
    print(len(mydata))
