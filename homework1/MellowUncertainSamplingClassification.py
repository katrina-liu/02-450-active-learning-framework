from sklearn.naive_bayes import GaussianNB

import Parser
from homework1.UncertainSamplingClassificationSimulation import UncertainSamplingClassificationSimulation
import random

class MellowUncertainSamplingClassification(UncertainSamplingClassificationSimulation):
    def increment_train(self):
        conf = self.confidence()
        incr_conf = sorted(enumerate(conf),key=lambda x:x[1])
        rand_idx = random.choice(incr_conf[:(len(incr_conf)+1)//2])[0]
        choice = self.unobserved.pop(rand_idx)
        self.train.append(choice)




if __name__ == '__main__':
    X_, y_ = Parser.parse_csv("../classification.csv", 2)
    cs = MellowUncertainSamplingClassification(X_, y_, GaussianNB(), 5)
    print(cs.cross_validation_on_train(5))
    print(cs.predict())
    print(cs.train)
    print(cs.unobserved)
    cs.increment_train()
    print(cs.train)





