import sklearn
from sklearn.ensemble import RandomForestClassifier

from classifiers.classification_model import ClassificationModel


class RandomForest(ClassificationModel):
    def __init__(self, training_samples, test_samples, training_labels, test_labels):
        super().__init__(training_samples, test_samples, training_labels, test_labels)
        self.initializer = RandomForestClassifier
        self.name = "random-forest"
        self.tune_parameter = "n_estimators"
