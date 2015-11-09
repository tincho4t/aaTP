from sklearn.linear_model import LogisticRegression
from Classifier import Classifier
from Constants import Constants

class LogisticRegression(Classifier):

    def __init__(self, X, Y):
        super(LogisticRegression, self).__init__(X, Y)
        self.classifier = LogisticRegression(random_state=Constants.RANDOM_STATE)
