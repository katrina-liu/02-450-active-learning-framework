"""An aggressive active learning model to train classifiers instrumented with the ability
to choose the next unobserved data by observing the distance-weighted uncertainty
to add to the training dataset.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB

from src import Parser
from src.ClassificationSimulation import ClassificationSimulation

__author__ = "Xiao(Katrina) Liu"
__credits__ = ["Xiao Liu"]
__email__ = "xiaol3@andrew.cmu.edu"


class DensityBasedSamplingSimulation(ClassificationSimulation):
    def __init__(self, X, y, base_learner, seed_num, inform_fn, sim_fn, beta):
        """
        Constructor of a density based sampling simulator
        :param X: Complete input X data
        :param y: Complete input label data
        :param base_learner: base learner for the simulation
        :param seed_num: number of initial random seeds in the training set
        :param inform_fn: function to measure informativeness of an instance
        :param sim_fn: function to measure similarity between two instances
        :param beta: importance factor of similarity
        """
        super().__init__(X, y, base_learner, seed_num)
        self.inform_fn = inform_fn
        self.sim_fn = sim_fn
        self.beta = beta
        self.sim_mat = [[self.sim_fn(x, x_) for x in X] for x_ in X]

    def compute_density(self, i):
        """
        Compute the weight for the i-th unobserved instance
        :param i: index of the instance in unobserved dataset
        :return: weight of the input
        """
        unobserved_density = [self.sim_mat[i][j] for j in self.unobserved]
        avg_density = sum(unobserved_density) / (len(self.unobserved))
        return avg_density ** self.beta

    def increment_train(self):
        """
        Move one instance from the unobserved dataset to the training dataset
        by computing the maximum weighted informativeness of each unobserved
        instance
        :return:
        """
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        test_inform = self.inform_fn(X_train, y_train, X_test,
                                     self.base_learner)
        weighted_inform = map(
            lambda i: self.compute_density(self.unobserved[i]) * test_inform[i],
            range(len(self.unobserved)))
        max_inform_index = max(enumerate(weighted_inform), key=lambda x: x[1])[
            0]
        choice = self.unobserved.pop(max_inform_index)
        self.train.append(choice)


if __name__ == "__main__":
    X_, y_ = Parser.parse_csv("../../data/classification.csv", 2)


    def least_confident(x_train, y_train, x_test, learner):
        learner.fit(x_train, y_train)
        test_proba = learner.predict_proba(x_test)
        return list(map(lambda x: 1 - max(x), test_proba))


    def sim_exp_euclidean_distance(x_1, x_2):
        diff_sum = 0
        for i in range(len(x_1)):
            diff_sum += (x_1[i] - x_2[i]) ** 2
        return 1 / (np.exp(diff_sum ** 0.5))


    cs = DensityBasedSamplingSimulation(X_, y_, GaussianNB(), 10,
                                        least_confident,
                                        sim_exp_euclidean_distance, 0.5)
    print(cs.train)
    cs.increment_train()
    print(cs.train)
