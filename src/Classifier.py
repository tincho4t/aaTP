# Class Classifier(path_to_model_directory=Null):
# init: model_1,model_2,model_n, ensemble_model, preprocessing_class_load_dataset()
# function train_model1:
    # dataset = preprocesclass.get_histogram()
    # train(dataset)
# function train_model2
# function train_modeln
# function predict(path_to_image_directory): 
    # preproceclass = new preprocessclase(path_to_image)
    # dataset_test_model_1 = preprocessclass.get_historgarm()
    # model_1.predict(dataset_test_model_1) 
# function save -> Save all sub models + model ensamble
# function load -> Load all sub models + model ensamble
import numpy as np
from sklearn.cross_validation import train_test_split

class Classifier(object):
    N_ESTIMATORS = 100
    RANDOM_STATE = 42

    def __init__(self, X, Y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.20, random_state=self.RANDOM_STATE)
        self.classifier = None
        self.prediction = None
        self.accuracy = None

    def fit(self):
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        self.prediction = self.classifier.predict(self.X_test)
        return self.prediction

    def score(self):
        return self.classifier.score(self.X_test, self.y_test)
