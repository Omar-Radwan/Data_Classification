import sklearn

from classifiers.classification_model import ClassificationModel


class DecisionTree(ClassificationModel):
    def __init__(self, training_samples, test_samples, training_labels, test_labels):
        super().__init__(training_samples, test_samples, training_labels, test_labels)
        self.initializer = sklearn.tree.DecisionTreeClassifier
        self.name = "decision-tree"
