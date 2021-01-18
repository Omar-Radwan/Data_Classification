import os

import numpy as np
from sklearn.model_selection import train_test_split

from classifiers.ada_boost import AdaBoost
from classifiers.classification_model import ClassificationModel
from classifiers.decision_tree import DecisionTree
from classifiers.k_nearest import KNearest
from classifiers.naive_bayes import NaiveBayes
from classifiers.random_forest import RandomForest

from misc.excel_writer import ExcelWriter
from misc.input_parser import InputParser
from misc.constants import *


def model_output(classification_model: ClassificationModel, tune_start, tune_end):
    classification_model.tune(tune_start, tune_end)
    classification_model.fit()
    classification_model.evaluate_model()
    excel_writer = ExcelWriter(classification_model.name)
    excel_writer.edit_sheet(classification_model.report_dict)


if __name__ == '__main__':
    if os.path.exists(WORK_BOOK_PATH):
        os.remove(WORK_BOOK_PATH)

    input_parser = InputParser()
    data_tuple = input_parser.get_samples_and_labels()

    samples, labels = np.array(data_tuple[0]), np.array(data_tuple[1])
    training_samples, test_samples, training_labels, test_labels = train_test_split(samples, labels,
                                                                                    test_size=TEST_SIZE,
                                                                                    random_state=0)

    model_output(NaiveBayes(training_samples, test_samples, training_labels, test_labels), None, None)
    model_output(KNearest(training_samples, test_samples, training_labels, test_labels), MIN_N_NEIGHBOUR,
                 MAX_N_NEIGHBOUR)
    model_output(RandomForest(training_samples, test_samples, training_labels, test_labels), MIN_N_ESTIMATE,
                 MAX_N_ESTIMATE)
    model_output(DecisionTree(training_samples, test_samples, training_labels, test_labels), None, None)
    model_output(AdaBoost(training_samples, test_samples, training_labels, test_labels), MIN_N_ESTIMATE, MAX_N_ESTIMATE)
