from sklearn.ensemble import RandomForestClassifier
from Classifier import Classifier
from Constants import Constants

class RandomForest(Classifier):

    def __init__(self, X, Y, n_estimators = 1000):
        super(RandomForest, self).__init__(X, Y)
        self.n_estimators = n_estimators
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=Constants.RANDOM_STATE)

