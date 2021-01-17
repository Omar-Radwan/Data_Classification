import sklearn
from sklearn.naive_bayes import GaussianNB

from classifiers.classification_model import ClassificationModel


class NaiveBayes(ClassificationModel):
    def __init__(self, training_samples, test_samples, training_labels, test_labels):
        super().__init__(training_samples, test_samples, training_labels, test_labels)

        self.name = "naive-bayes"
        self.initializer = GaussianNB
