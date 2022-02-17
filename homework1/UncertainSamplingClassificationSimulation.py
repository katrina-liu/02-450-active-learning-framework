from sklearn.naive_bayes import GaussianNB

import Parser
from ClassificationSimulation import ClassificationSimulation

class UncertainSamplingClassificationSimulation(ClassificationSimulation):
    def increment_train(self):
        """
        Add the most uncertain observation to the train data
        :return:
        """
        min_index = self.least_confident()
        choice = self.unobserved.pop(min_index)
        self.train.append(choice)

    def least_confident(self):
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        self.base_learner.fit(X_train,y_train)
        test_proba = self.base_learner.predict_proba(X_test)
        # choose based on minmax prediction proba
        min_proba = 1
        min_index = -1
        for i in range(len(X_test)):
            proba = max(test_proba[i])
            if proba < min_proba:
                min_proba = proba
                min_index = i
        # print(min_proba)
        return min_index


if __name__ == '__main__':
    X_, y_ = Parser.parse_csv("../classification.csv", 2)
    cs = UncertainSamplingClassificationSimulation(X_, y_, GaussianNB(), 5)
    print(cs.cross_validation_on_train(5))
    print(cs.predict())
    print(cs.train)
    print(cs.unobserved)
    cs.increment_train()
    print(cs.train)