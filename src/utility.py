import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def nearest_neighbor(point, candidate, num):
    """

    :param point: (x,y)
    :param candidate: a list of (x,y)
    :param num: select num nearest neighbors
    :return:
    """
    pass





if __name__ == "__main__":
    x = np.random.random(5)
    N = 2
    print(len(running_mean(x, N)))