# La idea de este archivo es tener un script pequenio que nos permita
# probar las distintas partes del codigo

import numpy as np
import matplotlib.pyplot as plt
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

import rlcompleter, readline
readline.parse_and_bind('tab:complete')


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

ip = ImagesProcessor()
images, y = ip.getImages('../imgs/test/medium/', training=True, size=(512, 512))

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.3)
X_train, y_train = ip.transformImages(X_train, y_train)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)

ensemble = Ensemble()
ensemble.fit_small(X_train, y_train)
X_validation_predictions = ensemble.predict_small(X_validation)
ensemble.fit_big(X_validation_predictions, y_validation)

X_test_predictions = ensemble.predict_small(X_test)
y_hat = ensemble.predict_big(X_test_predictions)
print(score(y_hat, y_test))

y_hat_voting = np.zeros((len(X_test)))
X_test_triple, _ = ip.transformImages(X_test)
X_test_predictions = ensemble.predict_small(X_test_triple)
y_hat = ensemble.predict_big(X_test_predictions)
for i in range(0, len(y_hat), 3):
    winner = find_majority(y_hat[i:i+3])
    y_hat_voting[i/3] = winner

print(score(y_hat_voting, y_test))



#plt.figure(figsize=(14.2, 10))
#for i, comp in enumerate(rbm1.components_):
    #plt.subplot(10, 10, i + 1)
    #plt.imshow(comp.reshape((56, 56)), cmap=plt.cm.gray_r, interpolation='nearest')
    #plt.xticks(())
    #plt.yticks(())
#
#plt.suptitle('100 components extracted by RBM', fontsize=16)
#plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#
#plt.show()

#images = ip.getImages()
#gimages = ip.getImagesWithGrayHistogramEqualized()
#getDarkPatternFeature = ip.getDarkPatternFeature() # Tarda!!!!

## Como mostrar una foto
#image = cv2.imread('../imgs/test/cat.0.jpg')
#cv2.imshow('color_image',image)
#cv2.waitKey(0)                 # Waits forever for user to press any key
#cv2.destroyAllWindows()        # Closes displayed windows

#image = ds[0]
#image = cv2.resize(image, (106, 106))
#
#cv2.imshow('color_image', image)
#cv2.waitKey(0)                 # Waits forever for user to press any key
#cv2.destroyAllWindows()        # Closes displayed windows

#image = cv2.imread('../imgs/test/cat.0.jpg', 0)
#ip.getDarkPattern(image)


#import mahotas
#radius = 5
#points = 12
#img = mahotas.imread("../imgs/test/cat.0.jpg", as_grey=True)
#mahotas.features.lbp(img, radius, points, ignore_zeros=False)


"""
    Dependencias:
        pip install mahotas
        sudo pip install imread
"""
"""
    Resultados: con RandomForest
        textures = ip.getTextureFeature(5,12)
        0.65336658354114718

        textures = ip.getTextureFeature(10,12)
        0.66583541147132175

        textures = ip.getTextureFeature(15,20)
        0.57356608478802995

        textures = ip.getTextureFeature(15,12)
        0.62094763092269323
"""