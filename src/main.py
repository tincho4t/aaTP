# La idea de este archivo es tener un script pequenio que nos permita
# probar las distintas partes del codigo

import numpy as np
import cv2
import os
from Constants import Constants
from ImagesProcessor import ImagesProcessor
from RandomForest import RandomForest
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import rlcompleter, readline
readline.parse_and_bind('tab:complete')

ip = ImagesProcessor('../imgs/test/medium/', training=True)

images = ip.getImages()


# def crossValidation(X, Y):
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=Constants.TEST_SIZE, random_state=Constants.RANDOM_STATE)
#     kf = kf = KFold(len(X_train), n_folds=30, shuffle=False, random_state=Constants.RANDOM_STATE)
#     for train_index, test_index in kf:
#         print("TRAIN:", train_index, "TEST:", test_index)


#ip = ImagesProcessor('../imgs/test/small/', training=True)

# textures = ip.getTextureFeature(5, 12)
# y = ip.getImagesClass()

# r = RandomForest(textures, y)
# r.fit()
# r.score()

# pca, norm, ds = ip.getPcaFeatures(10, (100, 100))
# y = ip.getImagesClass()
# pr = RandomForest(ds, y)
# pr.fit()
# pr.score()

# rbm, ds = ip.getBernulliRBM(20, (256, 256), n_iter=20, learning_rate=0.005)
# y = ip.getImagesClass()
# pr = RandomForest(ds, y)
# pr.fit()
# pr.score()


#images = ip.getImages()
#gimages = ip.getImagesWithGrayHistogramEqualized()
#getDarkPatternFeature = ip.getDarkPatternFeature() # Tarda!!!!

## Como mostrar una foto
#image = cv2.imread('../imgs/test/cat.0.jpg')
#cv2.imshow('color_image',image)
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