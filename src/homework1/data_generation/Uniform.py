"""Randomly generate uniformly distributed data with labels split by y=x.
"""

import csv

import numpy as np

__author__ = "Xiao(Katrina) Liu"
__credits__ = ["Xiao Liu"]
__email__ = "xiaol3@andrew.cmu.edu"


def generate_uniform_sample(r=(0, 10), dim=2, s=100, labels=(1, 2)):
    """
    Generate randomly uniformly distributed data points with labels split by
    the identity function.
    :param r: range of the sample value
    :param dim: dimension of the data
    :param s: number of samples to be generated
    :param labels: label set
    :return: concatenated data of generated X and y
    """
    samples = np.random.random_sample((s, dim))
    y = []
    for i in range(s):
        y.append([labels[0]] if samples[i][0] > samples[i][1] else [labels[1]])
    return np.concatenate((samples * (r[1] - r[0]) + r[0], y), axis=1)


def write_data_to_csv(filename, data, dim=2):
    writer = csv.writer(open(filename, "w", newline=""))
    header = ["x%d" % (d + 1) for d in range(dim)]
    header.append("y")
    writer.writerow(header)
    writer.writerows(data)


if __name__ == "__main__":
    d = generate_uniform_sample()
    write_data_to_csv("../../../data/uniform_0_10_100.csv", d)
