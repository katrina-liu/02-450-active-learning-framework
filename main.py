import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

import ClassificationSimulation
import Parser
import RegressionSimulation


def train(cycle, stop, simulator):
    """
    Perform training for a simulator
    :param cycle: number of cycles of simulation rounds
    :param stop: stopping criterion
    :param simulator: classification or regression
    :return: training data size, cross validation accuracy/mse, prediction accuracy/mse
    """
    cross_validation_acc = []
    prediction_acc = []
    train_size = []
    for _ in range(cycle):
        simulator.clear()
        cv_acc = []
        pred_acc = []
        train_size = []

        while simulator.train_size() <= stop:
            train_size.append(simulator.train_size())
            cv_acc.append(simulator.cross_validation_on_train(5))
            pred_acc.append(simulator.predict())
            simulator.increment_train()
        cross_validation_acc.append(cv_acc)
        prediction_acc.append(pred_acc)
    print(cross_validation_acc)
    print(prediction_acc)
    cross_validation_acc = np.array(cross_validation_acc)
    prediction_acc = np.array(prediction_acc)
    return train_size, cross_validation_acc, prediction_acc


if __name__ == "__main__":
    class_input_X, class_input_y = Parser.parse_csv("classification.csv", 2)
    stop = 50
    cycle = 10
    seed_num = 5
    train_size, cross_validation_acc, prediction_acc = train(cycle, stop,
                                                             ClassificationSimulation.ClassificationSimulation(
                                                                 class_input_X,
                                                                 class_input_y,
                                                                 GaussianNB(),
                                                                 seed_num))
    plt.figure()
    plt.plot(range(len(train_size)), train_size)
    plt.xlabel("Round")
    plt.ylabel("Training Data Size")
    plt.grid()
    # plt.title("Training Data Size vs Rounds of Classification Simulation")
    plt.savefig("images/cs_train_size.png")

    plt.figure()
    plt.errorbar(range(len(cross_validation_acc[0])),
                 np.mean(cross_validation_acc, axis=0),
                 yerr=np.std(cross_validation_acc, axis=0),
                 label="5-Fold Cross-validation", fmt='-o', capsize=3)
    plt.errorbar(range(len(prediction_acc[0])),
                 np.mean(prediction_acc, axis=0),
                 yerr=np.std(prediction_acc, axis=0),
                 label="Prediction of Unobserved", fmt='-o', capsize=3)
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig("images/cs_acc.png")

    regress_input_X, regress_input_y = Parser.parse_csv("regression.csv", 2)
    train_size, cross_validation_mse, prediction_mse = train(cycle, stop,
                                                             RegressionSimulation.RegressionSimulation(
                                                                 regress_input_X,
                                                                 regress_input_y,
                                                                 LinearRegression(),
                                                                 seed_num))
    print(cross_validation_mse, prediction_mse)

    plt.figure()
    plt.errorbar(range(len(cross_validation_mse[0])),
                 np.mean(cross_validation_mse, axis=0),
                 yerr=np.std(cross_validation_mse, axis=0),
                 label="5-Fold Cross-validation", fmt='-o', capsize=3)
    plt.errorbar(range(len(prediction_mse[0])),
                 np.mean(prediction_mse, axis=0),
                 yerr=np.std(prediction_mse, axis=0),
                 label="Prediction of Unobserved", fmt='-o', capsize=3)
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    plt.savefig("images/rs_acc.png")
