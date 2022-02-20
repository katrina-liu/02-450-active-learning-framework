"""Randomly generate clustered data with labels split by y=x.
"""

import numpy as np
from sklearn.datasets import make_blobs

from homework1.data_generation.Uniform import write_data_to_csv

__author__ = "Xiao(Katrina) Liu"
__credits__ = ["Xiao Liu"]
__email__ = "xiaol3@andrew.cmu.edu"


def generate_clustering_sample(r=(0, 10), dim=2, s=100, labels=(1, 2)):
    """
    Generate the data by cluster along the y=upper_range-x with label split by
    y = x
    :param r: range of generated data
    :param dim: dimension of the data
    :param s: number of samples to generate
    :param labels: label set
    :return: concatenated clustered data of X and y
    """
    sample_x, _ = make_blobs(s, centers=[(2.5, 7.5), (7.5, 2.5)],
                             n_features=dim, cluster_std=2.5)
    sample_y = []
    for i in range(s):
        sample_y.append(
            [labels[0]] if sample_x[i][0] > sample_x[i][1] else [labels[1]])

    d = np.concatenate((sample_x, sample_y), axis=1)
    np.random.shuffle(d)
    return d


if __name__ == "__main__":
    d = generate_clustering_sample()
    print(d)
    write_data_to_csv("../../../data/cluster_0_10_100.csv", d)
