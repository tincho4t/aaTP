import numpy as np
from sklearn.cross_validation import train_test_split
from Constants import Constants

class Classifier(object):

    def __init__(self, X, Y):
        self.X_train = X
        self.y_train = Y
        self.classifier = None
        self.accuracy = None

    def fit(self):
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.classifier.predict(X)
