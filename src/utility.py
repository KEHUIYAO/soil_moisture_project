import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)





if __name__ == "__main__":
    x = np.random.random(5)
    N = 2
    print(len(running_mean(x, N)))