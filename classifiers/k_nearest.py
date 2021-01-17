from sklearn.neighbors import KNeighborsClassifier

from classifiers.classification_model import ClassificationModel
from misc.constants import *
from sklearn.model_selection import cross_val_score


class KNearest(ClassificationModel):
    def __init__(self, training_samples, test_samples, training_labels, test_labels):
        super().__init__(training_samples, test_samples, training_labels, test_labels)
        self.initializer = KNeighborsClassifier
        self.name = "k-nearest-neighbour"
        self.tune_parameter = "n_neighbors"
