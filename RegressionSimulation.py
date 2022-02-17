import random

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import Parser


class RegressionSimulation:
    def __init__(self, X, y, base_learner, seed_num):
        self.X = X
        self.y = y
        self.train = []
        self.unobserved = []
        self.base_learner = base_learner
        self.seed_num = seed_num
        seeds = random.sample(range(len(self.X)), seed_num)
        # print(seeds)
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
        :return: mean squared error of cross validation
        """
        kf = KFold(n_splits=fold)
        cv_err = []
        # print(self.train)
        for train, test in kf.split(self.train):
            train_X = [self.X[self.train[i]] for i in train]
            train_y = [self.y[self.train[i]] for i in train]
            test_X = [self.X[self.train[i]] for i in test]
            test_y = [self.y[self.train[i]] for i in test]
            self.base_learner.fit(train_X, train_y)
            y_predict = self.base_learner.predict(test_X)
            for i in range(len(test)):
                cv_err.append(test_y[i] - y_predict[i])
        return (np.square(cv_err)).mean()

    def predict(self):
        """
        Train on training data and predict on unobserved data
        :return: mean square error of predicting unobserved data
        """
        total_err = []
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        y_test = [self.y[i] for i in self.unobserved]
        self.base_learner.fit(X_train, y_train)
        y_predict = self.base_learner.predict(X_test)
        for i in range(len(X_test)):
            total_err.append(y_test[i] - y_predict[i])
        return (np.square(total_err)).mean()

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
    X_, y_ = Parser.parse_csv("regression.csv", 2)
    cs = RegressionSimulation(X_, y_, LinearRegression(), 5)
    print(cs.cross_validation_on_train(5))
    print(cs.predict())
