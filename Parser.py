import csv

import numpy as np


def parse_csv(filename, x_dim):
    """
    Parse the data set from csv file
    :param filename: source file name
    :param x_dim: dimension of the x data
    :return: x data and y data
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        X = []
        y = []
        for row in reader:
            X.append([float(elem) for elem in row[:x_dim]])
            y.append(float(row[x_dim]))
        return np.array(X), np.array(y)


if __name__ == "__main__":
    X_, y_ = parse_csv("classification.csv", 2)
    print(X_, y_)
