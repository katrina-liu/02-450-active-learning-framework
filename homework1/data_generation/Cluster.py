import numpy as np
import math
from sklearn.datasets import make_blobs
from homework1.data_generation.Uniform import write_data_to_csv,generate_uniform_sample

def generate_clustering_sample(r=(0,10), dim=2, s=100,labels=(1,2)):
    sample_x, _ = make_blobs(s,centers=[(2.5,7.5),(7.5,2.5)],n_features=dim,cluster_std=2.5)
    sample_y = []
    for i in range(s):
        sample_y.append([labels[0]] if sample_x[i][0]>sample_x[i][1] else [labels[1]])

    d = np.concatenate((sample_x,sample_y),axis=1)
    np.random.shuffle(d)
    return d

if __name__ == "__main__":
    d = generate_clustering_sample()
    print(d)
    write_data_to_csv("cluster_0_10_100.csv",d)