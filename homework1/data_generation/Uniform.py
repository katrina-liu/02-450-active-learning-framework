import numpy as np
import csv


def generate_uniform_sample(r=(0,10), dim=2, s=100,labels=(1,2)):
    samples = np.random.random_sample((s,dim))
    y = []
    for i in range(s):
        y.append([labels[0]] if samples[i][0]>samples[i][1] else [labels[1]])
    return np.concatenate((samples*(r[1]-r[0])+r[0],y),axis=1)


def write_data_to_csv(filename, data, dim=2):
    writer = csv.writer(open(filename,"w",newline=""))
    header = ["x%d"%(d+1)for d in range(dim)]
    header.append("y")
    writer.writerow(header)
    writer.writerows(data)

if __name__ == "__main__":
    d = generate_uniform_sample()
    write_data_to_csv("uniform_0_10_100.csv",d)

