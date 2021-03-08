import numpy as np
import pandas as pd

from utility import nearest_neighbor


data = pd.read_csv("../../SMAP_Climate_In_Situ.csv")
# SMAP observations
data_smap = data.iloc[:946080,:]
data_smap['ind'] = np.arange(data_smap.shape[0])
# Insitu dataset
data_insitu = data.iloc[946080:,:]
data_insitu['ind'] = np.arange(data_insitu.shape[0])



# create a map from point id to insitu id
point_id = data_smap[~data_smap.duplicated('POINTID')].loc[:,['POINTID','x','y','Date','ind']]


insitu_id = data_insitu[~data_insitu.duplicated('POINTID')].loc[:,['POINTID','x','y','Date','ind']]

#print(data_smap.shape)
#print(data_insitu.shape)

point_id['new_ind'] = np.arange(point_id.shape[0])
insitu_id['new_ind'] = np.arange(insitu_id.shape[0])


mapping = {}
candidate = insitu_id.iloc[:, 1:3].values
for i in range(point_id.shape[0]):
    point = point_id.iloc[i,1:3].values
    ind, _ = nearest_neighbor(point, candidate)
    mapping[point_id.iloc[i,0]] = insitu_id.iloc[ind,0]

print(mapping)

candidate_point_id = point_id['POINTID'].values

for point_id_index in range(len(candidate_point_id)):


    id = candidate_point_id[point_id_index]

    point_id_new_ind = point_id['new_ind'][point_id['POINTID'] == id]

    start = point_id['ind'].iloc[point_id_new_ind].values[0]

    if point_id_new_ind.values < point_id.shape[0] - 1:
        end = point_id['ind'].iloc[point_id_new_ind+1].values[0]
    else:
        end = data_smap.shape[0]

    id = mapping[id]

    insitu_id_new_ind = insitu_id['new_ind'][insitu_id['POINTID'] == id]
    match_start = insitu_id['ind'].iloc[insitu_id_new_ind].values[0]

    if insitu_id_new_ind.values < insitu_id.shape[0] - 1:
        match_end = insitu_id['ind'].iloc[insitu_id_new_ind+1].values[0]
    else:
        match_end = data_insitu.shape[0]



    start_date = data_smap.iloc[start, 4]
    print(match_start, match_end)
    flag = False
    for k in range(match_start, match_end):
        if data_insitu.iloc[k, 4] == start_date:
            flag = True
            break
    if flag:

        data_smap.iloc[start:end, 11] = data_insitu.iloc[k:(k+end-start),12]
    else:
        print("no match")

    print(point_id_index)
    #print(start,end)
    #print(match_start, match_end)

data_smap.to_csv("Insitu_gap_filling_data.csv")

