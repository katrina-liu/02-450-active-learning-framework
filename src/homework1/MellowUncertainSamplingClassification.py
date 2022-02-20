"""An mellow active learning model to train classifiers instrumented with the
ability to choose the next unobserved data which the current model is relatively
uncertain about to add to the training dataset.
"""

import random

from sklearn.naive_bayes import GaussianNB

from homework1.UncertainSamplingClassificationSimulation import \
    UncertainSamplingClassificationSimulation
from src import Parser

__author__ = "Xiao(Katrina) Liu"
__credits__ = ["Xiao Liu"]
__email__ = "xiaol3@andrew.cmu.edu"


class MellowUncertainSamplingClassification(
    UncertainSamplingClassificationSimulation):
    def increment_train(self):
        """
        Move one instance from the unobserved dataset to the training dataset
        by randomly picking one of the instances that is in the top half in the
        informativeness, i.e. the reverse confidence level.
        :return:
        """
        conf = self.confidence()
        incr_conf = sorted(enumerate(conf), key=lambda x: x[1])
        rand_idx = random.choice(incr_conf[:(len(incr_conf) + 1) // 2])[0]
        choice = self.unobserved.pop(rand_idx)
        self.train.append(choice)


if __name__ == '__main__':
    X_, y_ = Parser.parse_csv("../../data/classification.csv", 2)
    cs = MellowUncertainSamplingClassification(X_, y_, GaussianNB(), 5)
    print(cs.cross_validation_on_train(5))
    print(cs.predict())
    print(cs.train)
    print(cs.unobserved)
    cs.increment_train()
    print(cs.train)
