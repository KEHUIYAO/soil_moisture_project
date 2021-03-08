import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def nearest_neighbor(point, candidates):
    """

    :param point: (x,y)
    :param candidate: a list of (x,y)

    :return:
    """

    point = np.array(point, dtype=np.float32)
    candidates = np.array(candidates, dtype=np.float32)
    point = np.ones_like(candidates) * point
    distance = np.linalg.norm(point-candidates, axis = 1)
    min_distance_ind = np.argmin(distance)
    min_distance = distance[min_distance_ind]

    return min_distance_ind, min_distance







if __name__ == "__main__":
    x = np.random.random(5)
    N = 2
    print(len(running_mean(x, N)))

    a = np.array([0,0])
    b = np.array(([[1,1],[2,2]]))
    print(nearest_neighbor(a,b))