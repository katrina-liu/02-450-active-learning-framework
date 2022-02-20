"""An aggressive active learning model to train classifiers instrumented with the ability
to choose the next unobserved data where the current model is the least confident
in making a predicting to add to the training dataset.
"""

from sklearn.naive_bayes import GaussianNB

from src import Parser
from src.ClassificationSimulation import ClassificationSimulation

__author__ = "Xiao(Katrina) Liu"
__credits__ = ["Xiao Liu"]
__email__ = "xiaol3@andrew.cmu.edu"


class UncertainSamplingClassificationSimulation(ClassificationSimulation):
    def increment_train(self):
        """
        Add the most uncertain observation to the train data
        :return:
        """
        min_index = self.least_confident()
        choice = self.unobserved.pop(min_index)
        self.train.append(choice)

    def confidence(self):
        """
        Compute the confidence level by the probability of the most probable
        label.
        :return: The maximum probability in prediction for each unobserved
        instance.
        """
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        self.base_learner.fit(X_train, y_train)
        test_proba = self.base_learner.predict_proba(X_test)
        return map(max, test_proba)

    def least_confident(self):
        """
        Find the index of the least confident instance in the unobserved set.
        :return: The index of the least confident instance in the unobserved
        set.
        """
        conf = self.confidence()
        # choose based on minmax prediction proba
        return min(enumerate(conf), key=lambda x: x[1])[0]


if __name__ == '__main__':
    X_, y_ = Parser.parse_csv("../../data/classification.csv", 2)
    cs = UncertainSamplingClassificationSimulation(X_, y_, GaussianNB(), 5)
    print(cs.cross_validation_on_train(5))
    print(cs.predict())
    print(cs.train)
    print(cs.unobserved)
    cs.increment_train()
    print(cs.train)
