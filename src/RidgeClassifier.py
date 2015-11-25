from sklearn import linear_model
from Classifier import Classifier
from Constants import Constants

class RidgeClassifier(Classifier):

    def __init__(self, X, Y, alpha=2):
        super(RidgeClassifier, self).__init__(X, Y)
        self.alpha = alpha
        self.classifier = linear_model.RidgeClassifier(alpha=self.alpha)

