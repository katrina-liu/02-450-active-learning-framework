"""A framework for training classification model instrumented with the ability
to randomly choose the next unobserved data to add to the training dataset.
"""

import random

import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

import Parser

__author__ = "Xiao(Katrina) Liu"
__credits__ = ["Xiao Liu"]
__email__ = "xiaol3@andrew.cmu.edu"


class ClassificationSimulation:
    def __init__(self, X, y, base_learner, seed_num):
        self.X = X
        self.y = y
        self.train = []
        self.unobserved = []
        self.seed_num = seed_num
        seeds = random.sample(range(len(self.X)), seed_num)
        self.base_learner = base_learner
        for i in range(len(X)):
            if i in seeds:
                self.train.append(i)
            else:
                self.unobserved.append(i)

    def clear(self):
        """
        Re-initialize the simulator with new seeds
        """
        self.train = []
        self.unobserved = []
        seeds = random.sample(range(len(self.X)), self.seed_num)
        # print(seeds)
        for i in range(len(self.X)):
            if i in seeds:
                self.train.append(i)
            else:
                self.unobserved.append(i)

    def cross_validation_on_train(self, fold):
        """
        Perform cross validation on training dataset
        :param fold: number of folds for cross validation
        :return: accuracy of cross validation
        """
        kf = KFold(n_splits=fold)
        cv_err = 0
        # print(self.train)
        for train, test in kf.split(self.train):
            train_X = [self.X[self.train[i]] for i in train]
            train_y = [self.y[self.train[i]] for i in train]
            test_X = [self.X[self.train[i]] for i in test]
            test_y = [self.y[self.train[i]] for i in test]
            self.base_learner.fit(train_X, train_y)
            y_predict = self.base_learner.predict(test_X)
            for i in range(len(test)):
                if test_y[i] != y_predict[i]:
                    cv_err += 1
        return 1 - cv_err / len(self.train)

    def predict(self):
        """
        Train on training data and predict on unobserved data
        :return: accuracy of predicting unobserved data
        """
        total_err = 0
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        y_test = [self.y[i] for i in self.unobserved]
        self.base_learner.fit(X_train, y_train)
        y_predict = self.base_learner.predict(X_test)
        for i in range(len(X_test)):
            if y_test[i] != y_predict[i]:
                total_err += 1
        return 1 - total_err / len(self.unobserved)

    def increment_train(self):
        """
        Add one random observation ot the training data
        """
        choice = np.random.choice(self.unobserved)
        self.unobserved.remove(choice)
        self.train.append(choice)

    def train_size(self):
        """
        Get training data size
        :return: training data size
        """
        return len(self.train)


if __name__ == "__main__":
    X_, y_ = Parser.parse_csv("../data/classification.csv", 2)
    cs = ClassificationSimulation(X_, y_, GaussianNB(), 5)
    print(cs.cross_validation_on_train(5))
    print(cs.predict())
