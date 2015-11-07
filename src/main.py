# La idea de este archivo es tener un script pequenio que nos permita
# probar las distintas partes del codigo

import numpy as np
import cv2
import os
from ImagesProcessor import ImagesProcessor
from RandomForest import RandomForest

import rlcompleter, readline
readline.parse_and_bind('tab:complete')

ip = ImagesProcessor('../imgs/test/medium/', training=True)
#ip = ImagesProcessor('../imgs/test/small/', training=True)

textures = ip.getTextureFeature(5, 12)
y = ip.getImagesClass()

r = RandomForest(textures, y)
r.fit()
r.score()

pca, norm, ds = ip.getPcaFeatures(10, (100, 100))
y = ip.getImagesClass()
pr = RandomForest(ds, y)
pr.fit()
pr.score()

rbm, ds = ip.getBernulliRBM(20, (256, 256), n_iter=20, learning_rate=0.005)
y = ip.getImagesClass()
pr = RandomForest(ds, y)
pr.fit()
pr.score()

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