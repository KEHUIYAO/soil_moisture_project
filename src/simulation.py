import numpy as np
import pandas as pd

def simulation_1(N, missing_rate = 0):
    "generate y(t) = x1(t) + x2(t-1) + epsilon(t), where epsilon(t) is auto-regressive"

    # the range of seq_len
    seq_len_range = [200, 250]

    # object to store the simulated data
    data = pd.DataFrame({'POINTID':[], 'Date':[], 'x1':[], 'x2':[], 'x3':[], 'y':[], 'SMAP_1km':[]})

    # variance of the error
    sigma2 = 1

    # begin generating the data
    for i in range(N):
        # generate a random number as the current seqence length
        cur_seq_len = np.random.random_integers(low=seq_len_range[0], high=seq_len_range[1])
        y = np.empty(cur_seq_len)
        smap_1km = np.zeros(cur_seq_len)
        x1 = np.empty(cur_seq_len)
        x2 = np.empty(cur_seq_len)
        x3 = np.ones(cur_seq_len) * np.random.uniform(0,1,1)
        point_id = np.ones(cur_seq_len) * i
        date = np.repeat('20101020', cur_seq_len)
        epsilon = np.empty(cur_seq_len)

        # y[1], y[2]
        epsilon[0] = np.random.normal(0, 0.5, 1)
        x1[0] = np.random.uniform(2, 3, 1)
        x2[0] = np.random.uniform(0, 1, 1)
        y[0] = x1[0] + x3[0] + epsilon[0]

        epsilon[1] = epsilon[0] + np.random.normal(0, 0.5, 1)
        x1[1] = np.random.uniform(2, 3, 1)
        x2[1] = np.random.uniform(0, 1, 1)
        y[1] = x1[1] + x2[0] + x3[0] + epsilon[1]

        # after y[2]
        for j in range(2, cur_seq_len):
            epsilon[j] = 0.8 * epsilon[j-1] + 0.2 * epsilon[j-2] + np.random.normal(0, 0.5, 1)
            x1[j] = np.random.uniform(2, 3, 1)
            x2[j] = np.random.uniform(0, 1, 1)
            y[j] = x1[j] + 0.6 * x2[j - 1] + 0.4 * x2[j - 2] + x3[0] + epsilon[j]

            # make some smap_1km values missing
            if np.random.uniform(low=0, high=1, size=1) < missing_rate:
                smap_1km[j] = 0
            else:
                smap_1km[j] = y[j]

        cur_data = pd.DataFrame({'POINTID':point_id, 'Date':date, 'x1':x1, 'x2':x2, 'x3':x3, 'y':y, 'SMAP_1km':smap_1km})

        data = pd.concat([data, cur_data])




    return data

if __name__ == "__main__":
    data = simulation_1(N=1000, missing_rate=0.1)
    data.to_csv('../../simulation_data.csv')