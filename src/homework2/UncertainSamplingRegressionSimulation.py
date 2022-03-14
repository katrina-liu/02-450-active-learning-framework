from src.RegressionSimulation import RegressionSimulation


class UncertainSamplingRegressionSimulation(RegressionSimulation):
    def increment_train(self):
        """
        Add the most uncertain observation to the train data
        :return:
        """
        min_index = self.highest_variance()
        choice = self.unobserved.pop(min_index)
        self.train.append(choice)

    def variance(self):
        """
        Compute the variance/squared of the error of each instance.
        :return: The variance for each unobserved
        instance.
        """
        X_train = [self.X[i] for i in self.train]
        y_train = [self.y[i] for i in self.train]
        X_test = [self.X[i] for i in self.unobserved]
        y_test = [self.y[i] for i in self.unobserved]
        self.base_learner.fit(X_train, y_train)
        cv_err = []
        y_predict = self.base_learner.predict(X_test)
        for i in range(len(X_test)):
            cv_err.append((y_test[i] - y_predict[i]) ** 2)

        return cv_err

    def highest_variance(self):
        """
        Find the index of the highest variance instance in the unobserved set based on the current model.
        :return: The index of the highest variance instance in the unobserved
        set.
        """
        var = self.variance()
        # choose based on minmax prediction proba
        return max(enumerate(var), key=lambda x: x[1])[0]
