# La idea de este archivo es tener un script pequenio que nos permita
# probar las distintas partes del codigo

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
import numpy as np
# import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import numpy as np
import cv2
import os
from Constants import Constants
from ImagesProcessor import ImagesProcessor
from RandomForest import RandomForest
from LogisticRegression import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from Ensemble import Ensemble
import gc

import rlcompleter, readline
readline.parse_and_bind('tab:complete')

import cPickle

def find_majority(k):
    myMap = {}
    maximum = (0, 0)
    for n in k:
        if n in myMap:
            myMap[n] += 1
        else:
            myMap[n] = 1
        if myMap[n] > maximum[1]:
            maximum = (n, myMap[n])
    return maximum[0]


def score(y_hat, y_test):
    return(1-sum(abs(y_hat-y_test))*1.0/len(y_test))


def run_kfold(method, kf, X, y, text, transformer=None):
    accuracy = 0
    fold = 0
    print("Running "+str(text))
    for train_index, test_index in kf:
        print("Starting fold "+str(fold))
        fold += 1
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        if transformer is not None:
            t = transformer.fit(X_train)
            X_train = t.transform(X_train)
            X_test = t.transform(X_test)
        if method == "rf":
            clf = RandomForest(X_train, y_train, n_estimators=1000)
            clf.fit()
        elif method == "lr":
            clf = linear_model.RidgeClassifier(alpha=2)
            clf.fit(X_train, y_train)
        elif method == "ex":
            clf = ExtraTreesClassifier(n_estimators=2000)
            clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        accuracy += score(y_hat, y_test)
    return(accuracy*1.0/len(kf))

def printResult(names, y, confidence):
    with open('grupoEitanTincho.txt', 'wb') as f:
        for i in range(0,len(y)):
            item_id = names[i].split('.')[0]
            result = None
            if(y[i] == 0):
                result = 1
            else:
                result = 0
            conf = confidence[i][y[i]]
            f.write("%s,%d,%f\n"%(item_id, result, conf))


ip = ImagesProcessor()
images, y = ip.getImages('../imgs/test/dataset/', size=None, training=False)

# Esto es lo que hay que usar para predecir el resultado final
if True:
    ensemble = Ensemble()
    ensemble.load()
    X_predictions = ensemble.predict_small(images)
    y_hat = ensemble.predict_big(X_predictions)
    confidence = ensemble.ensemble_logistic_regression.predict_proba(X_predictions)
    printResult(y, y_hat, confidence)
    #score(y_hat, y)


# Esto es lo que hay que usar para calcular al regression lineal y gurdarla
if False:
    ensemble = Ensemble()
    ensemble.load()
    X_validation_predictions = ensemble.predict_small(images)
    ensemble.fit_big(X_validation_predictions, y)
    f = file("./ensemble_logistic_regression", 'wb')
    cPickle.dump(ensemble.ensemble_logistic_regression, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

