import random

import matplotlib.pyplot as plt
from pip._vendor.pep517.dirtools import mkdir_p
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from misc.constants import *


class ClassificationModel:
    def __init__(self, training_samples, test_samples, training_labels, test_labels):
        self.training_samples = training_samples
        self.test_samples = test_samples
        self.training_labels = training_labels
        self.test_labels = test_labels
        self.predicted = []
        self.predicted = None
        self.classifier = None
        self.name = ""
        self.initializer = None
        self.tune_parameter = None
        self.report_dict = {F1_SCORE: {MICRO: 1, WEIGHTED: 2, MACRO: 3},
                            RECALL_SCORE: {MICRO: 1, WEIGHTED: 2, MACRO: 3},
                            PRECISION_SCORE: {MICRO: 1, WEIGHTED: 2, MACRO: 3},
                            SCORE: 0,
                            CONFUSION_MATRIX: [[]]}

    def tune(self, start=None, end=None):
        if self.tune_parameter is None:
            self.classifier = self.initializer()
            return
        scores = []
        maxi = (0, 0)
        print(f'tunning {self.name}')
        for parameter in range(start, end + 1):
            cur = self.initializer(**{self.tune_parameter: parameter})
            score = cross_val_score(cur, self.training_samples, self.training_labels).mean()
            scores.append(score)
            print(f'{self.tune_parameter}={parameter}')
            maxi = max(maxi, (score, parameter))

        self.plot([i + 1 for i in range(end - start + 1)], scores, self.tune_parameter, MEAN_VALUE_OF_CROSS_VALIDATION,
                  f'{self.name} tuning')
        self.classifier = self.initializer(**{self.tune_parameter: maxi[1]})

    def evaluate_model(self):

        self.report_dict[F1_SCORE][MICRO] = f1_score(self.test_labels, self.predicted, average=MICRO)
        self.report_dict[F1_SCORE][MACRO] = f1_score(self.test_labels, self.predicted, average=MACRO)
        self.report_dict[F1_SCORE][WEIGHTED] = f1_score(self.test_labels, self.predicted, average=WEIGHTED)
        self.report_dict[RECALL_SCORE][MICRO] = recall_score(self.test_labels, self.predicted, average=MICRO)
        self.report_dict[RECALL_SCORE][MACRO] = recall_score(self.test_labels, self.predicted, average=MACRO)
        self.report_dict[RECALL_SCORE][WEIGHTED] = recall_score(self.test_labels, self.predicted, average=WEIGHTED)
        self.report_dict[PRECISION_SCORE][MICRO] = precision_score(self.test_labels, self.predicted, average=MICRO)
        self.report_dict[PRECISION_SCORE][MACRO] = precision_score(self.test_labels, self.predicted, average=MACRO)
        self.report_dict[PRECISION_SCORE][WEIGHTED] = precision_score(self.test_labels, self.predicted,
                                                                      average=WEIGHTED)
        self.report_dict[SCORE] = self.classifier.score(self.test_samples, self.test_labels)
        self.report_dict[CONFUSION_MATRIX] = confusion_matrix(self.test_labels, self.predicted)

    def plot(self, range, list, x_axis_label, y_axis_label, file_name=str(random.random())):

        plt.plot(range, list)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.savefig(self.__create_req_directories() + f'/{file_name}.png')
        plt.cla()
        plt.clf()
        # plt.close()

    def __create_req_directories(self):
        output_dir = f'./{OUTPUT_DIRECTORY}'
        mkdir_p(output_dir)
        output_dir += f'/{self.name}'
        mkdir_p(output_dir)
        return output_dir

    def fit(self):
        self.predicted = self.classifier.fit(self.training_samples, self.training_labels).predict(
            self.test_samples)