from load_data import SoilMoistureDataset
import numpy as np
import torch
import pandas as pd
def split_data(csv_file, training_data_proportion):
    """
    split the data into training data and testing data
    :param csv_file:
    :param training_data_proportion:
    :return:
    """


    data = pd.read_csv(csv_file)
    data['formatted_date'] = pd.to_datetime(data.Date, format='%Y%m%d')
    data['year'] = data['formatted_date'].dt.year
    data['index'] = list(range(data.shape[0]))
    ind_list = data['index'][~data.duplicated(subset=['POINTID', 'year'])]
    ind_list = list(ind_list) + [data.shape[0]]

    training_set_ind = []
    testing_set_ind = []
    for i in range(len(ind_list)-1):
        if np.random.rand(1) < training_data_proportion:
            training_set_ind += list(range(ind_list[i],ind_list[i+1]))
        else:
            testing_set_ind += list(range(ind_list[i], ind_list[i+1]))


    training_data = data.iloc[training_set_ind,:]
    testing_data = data.iloc[testing_set_ind,:]

    return training_data, testing_data



if __name__ == '__main__':


    training_data, testing_data = split_data("../../SMAP_Climate_In_Situ_TxSON.csv",0.7)

    training_data.to_csv("../../SMAP_Climate_In_Situ_TxSON_training_data.csv")
    testing_data.to_csv("../../SMAP_Climate_In_Situ_TxSON_testing_data.csv")
    #
    #
    # training_data, testing_data = split_data("../../SMAP_Climate_In_Situ_Kenaston.csv",0.7)
    #
    # training_data.to_csv("../../SMAP_Climate_In_Situ_Kenaston_training_data.csv")
    # testing_data.to_csv("../../SMAP_Climate_In_Situ_Kenaston_testing_data.csv")

