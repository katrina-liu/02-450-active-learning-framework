from sklearn.naive_bayes import GaussianNB

import Parser
from ClassificationSimulation import ClassificationSimulation

class ExpectedErrorReductionSimulation(ClassificationSimulation):
    def increment_train(self):
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        self.base_learner.fit(X_train,y_train)
        test_proba = self.base_learner.predict_proba(X_test)
        loss = []
        for i in range(len(self.unobserved)):
            loss_i = 0
            x_i = X_test[i]
            proba = test_proba[i]
            new_test = []
            for j in range(len(self.unobserved)):
                if j != i:
                    new_test.append(X_test[j])
            new_x_train = X_train + [x_i]
            for j in range(len(proba)):
                p_i = proba[j]
                y_i = self.base_learner.classes_[j]
                new_y_train = y_train + [y_i]
                self.base_learner.fit(new_x_train,new_y_train)
                probas = self.base_learner.predict_proba(new_test)
                loss_i += p_i*(sum(map(lambda x:1-max(x),probas)))
            loss.append((i,loss_i))
        print(loss)
        min_index = min(loss,key=lambda x:x[1])[0]
        self.train.append(self.unobserved.pop(min_index))




if __name__=="__main__":
    X_, y_ = Parser.parse_csv("../classification.csv", 2)
    cs = ExpectedErrorReductionSimulation(X_,y_,GaussianNB(),5)
    for _ in range(5):
        cs.increment_train()
        print(cs.train)