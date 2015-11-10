from Classifier import Classifier
from Constants import Constants

from sklearn.svm import SVC

class SVM(Classifier):

    def __init__(self, X, Y, kernel='linear', C=1, gamma = 'auto'):
        super(SVM, self).__init__(X, Y)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma # Por el momento no hago nada con este parametro
        self.classifier = SVC(kernel=self.kernel, C=self.C, random_state=Constants.RANDOM_STATE)

