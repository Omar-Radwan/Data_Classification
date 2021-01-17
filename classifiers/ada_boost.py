import sklearn

from classifiers.classification_model import ClassificationModel


class AdaBoost(ClassificationModel):
    def __init__(self, training_samples, test_samples, training_labels, test_labels):
        super().__init__(training_samples, test_samples, training_labels, test_labels)
        from sklearn.ensemble import AdaBoostClassifier
        self.initializer = AdaBoostClassifier
        self.name = "ada-boost"
        self.tune_parameter = "n_estimators"
