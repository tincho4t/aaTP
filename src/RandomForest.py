from sklearn.ensemble import RandomForestClassifier
from Classifier import Classifier

class RandomForest(Classifier):

    def __init__(self, X, Y):
        super(RandomForest, self).__init__(X, Y)
        self.classifier = RandomForestClassifier(n_estimators=self.N_ESTIMATORS, random_state=42)

