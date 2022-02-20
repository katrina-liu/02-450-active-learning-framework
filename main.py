import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from homework1.UncertainSamplingClassificationSimulation import UncertainSamplingClassificationSimulation
from homework1.DensityBasedSamplingSimulation import DensityBasedSamplingSimulation
from homework1.ExpectedErrorReductionSimulation import ExpectedErrorReductionSimulation
from homework1.MellowUncertainSamplingClassification import MellowUncertainSamplingClassification

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
    class_input_X,class_input_y = Parser.parse_csv("homework1/data_generation/uniform_0_10_100.csv",2)
    # class_input_X, class_input_y = Parser.parse_csv(
    #     "homework1/data_generation/cluster_0_10_100.csv", 2)
    stop = 50
    cycle = 10
    seed_num = 5
    # homework 0 plot
    # train_size, cross_validation_acc, prediction_acc = train(cycle, stop,
    #                                                          ClassificationSimulation.ClassificationSimulation(
    #                                                              class_input_X,
    #                                                              class_input_y,
    #                                                              GaussianNB(),
    #                                                              seed_num))
    plt.figure()
    # plt.plot(range(len(train_size)), train_size)
    # plt.xlabel("Round")
    # plt.ylabel("Training Data Size")
    # plt.grid()
    # # plt.title("Training Data Size vs Rounds of Classification Simulation")
    # plt.savefig("images/cs_train_size.png")
    #
    # plt.figure()
    # plt.errorbar(range(len(cross_validation_acc[0])),
    #              np.mean(cross_validation_acc, axis=0),
    #              yerr=np.std(cross_validation_acc, axis=0),
    #              label="5-Fold Cross-validation", fmt='-o', capsize=3)
    # plt.errorbar(range(len(prediction_acc[0])),
    #              np.mean(prediction_acc, axis=0),
    #              yerr=np.std(prediction_acc, axis=0),
    #              label="Random Selection", fmt='-o', capsize=3)
    # plt.legend()
    # plt.xlabel("Round")
    # plt.ylabel("Accuracy")
    # plt.grid()
    # plt.savefig("images/cs_acc.png")
    #
    # regress_input_X, regress_input_y = Parser.parse_csv("regression.csv", 2)
    # train_size, cross_validation_mse, prediction_mse = train(cycle, stop,
    #                                                          RegressionSimulation.RegressionSimulation(
    #                                                              regress_input_X,
    #                                                              regress_input_y,
    #                                                              LinearRegression(),
    #                                                              seed_num))
    # print(cross_validation_mse, prediction_mse)
    #
    # plt.figure()
    # plt.errorbar(range(len(cross_validation_mse[0])),
    #              np.mean(cross_validation_mse, axis=0),
    #              yerr=np.std(cross_validation_mse, axis=0),
    #              label="5-Fold Cross-validation", fmt='-o', capsize=3)
    # plt.errorbar(range(len(prediction_mse[0])),
    #              np.mean(prediction_mse, axis=0),
    #              yerr=np.std(prediction_mse, axis=0),
    #              label="Prediction of Unobserved", fmt='-o', capsize=3)
    # plt.legend()
    # plt.xlabel("Round")
    # plt.ylabel("Mean Squared Error")
    # plt.grid()
    # plt.savefig("images/rs_acc.png")
    train_size, cross_validation_acc, prediction_acc = train(cycle, stop,
                                                             UncertainSamplingClassificationSimulation(
                                                                 class_input_X,
                                                                 class_input_y,
                                                                 GaussianNB(),
                                                                 seed_num))
    # plt.figure()
    # plt.errorbar(range(len(cross_validation_acc[0])),
    #              np.mean(cross_validation_acc, axis=0),
    #              yerr=np.std(cross_validation_acc, axis=0),
    #              label="5-Fold Cross-validation", fmt='-o', capsize=3)
    plt.errorbar(range(len(prediction_acc[0])),
                 np.mean(prediction_acc, axis=0),
                 yerr=np.std(prediction_acc, axis=0),
                 label="Uncertain Sampling", fmt='-o', capsize=3)
    # # plt.legend()
    # # plt.xlabel("Round")
    # # plt.ylabel("Accuracy")
    # # plt.grid()
    # # plt.show()
    # # plt.savefig("images/uscs_acc.png")
    #
    #
    def least_confident(x_train, y_train, x_test, learner):
        learner.fit(x_train, y_train)
        test_proba = learner.predict_proba(x_test)
        return list(map(lambda x: 1-max(x),test_proba))
    def sim_exp_euclidean_distance(x_1,x_2):
        diff_sum = 0
        for i in range(len(x_1)):
            diff_sum += (x_1[i]-x_2[i])**2
        return 1/(np.exp(diff_sum**0.5))

    train_size, cross_validation_acc, prediction_acc = train(cycle, stop,
                                                             DensityBasedSamplingSimulation(class_input_X,
                                                                 class_input_y,
                                                                 GaussianNB(),
                                                                 seed_num,least_confident,sim_exp_euclidean_distance,1))
    # plt.figure()
    # plt.errorbar(range(len(cross_validation_acc[0])),
    #              np.mean(cross_validation_acc, axis=0),
    #              yerr=np.std(cross_validation_acc, axis=0),
    #              label="5-Fold Cross-validation", fmt='-o', capsize=3)
    plt.errorbar(range(len(prediction_acc[0])),
                 np.mean(prediction_acc, axis=0),
                 yerr=np.std(prediction_acc, axis=0),
                 label="Density Based Sampling", fmt='-o', capsize=3)
    # plt.legend()
    # plt.xlabel("Round")
    # plt.ylabel("Accuracy")
    # plt.grid()
    # #plt.show()
    # plt.savefig("images/dbscs_acc.png")

    # _,_,prediction_acc = train(cycle,stop,ExpectedErrorReductionSimulation(class_input_X,class_input_y,GaussianNB(),seed_num))
    # # plt.figure()
    # # plt.errorbar(range(len(cross_validation_acc[0])),
    # #              np.mean(cross_validation_acc, axis=0),
    # #              yerr=np.std(cross_validation_acc, axis=0),
    # #              label="5-Fold Cross-validation", fmt='-o', capsize=3)
    # plt.errorbar(range(len(prediction_acc[0])),
    #              np.mean(prediction_acc, axis=0),
    #              yerr=np.std(prediction_acc, axis=0),
    #              label="Expected Error Reduction", fmt='-o', capsize=3)
    # train_size, cross_validation_acc, prediction_acc = train(cycle, stop,
    #                                                          MellowUncertainSamplingClassification(
    #                                                              class_input_X,
    #                                                              class_input_y,
    #                                                              GaussianNB(),
    #                                                              seed_num))
    # plt.errorbar(range(len(prediction_acc[0])),
    #              np.mean(prediction_acc, axis=0),
    #              yerr=np.std(prediction_acc, axis=0),
    #              label="Mellow Uncertain Sampling", fmt='-o', capsize=3)
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.show()
    plt.savefig("hw1_part3_1.png")