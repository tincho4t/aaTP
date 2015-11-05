# La idea de este archivo es tener un script pequenio que nos permita
# probar las distintas partes del codigo

import numpy as np
import cv2
import os
from ImagesProcessor import ImagesProcessor

ip = ImagesProcessor('../imgs/test/', training=True)

images = ip.getImages()
gimages = ip.getImagesWithGrayHistogramEqualized()
#getDarkPatternFeature = ip.getDarkPatternFeature() # Tarda!!!!

## Como mostrar una foto
#image = cv2.imread('../imgs/test/cat.0.jpg')
#cv2.imshow('color_image',image)
#cv2.waitKey(0)                 # Waits forever for user to press any key
#cv2.destroyAllWindows()        # Closes displayed windows

image = cv2.imread('../imgs/test/cat.0.jpg', 0)
ip.getDarkPattern(image)


import mahotas
radius = 5
points = 12
img = mahotas.imread("../imgs/test/cat.0.jpg", as_grey=True)
mahotas.features.lbp(img, radius, points, ignore_zeros=False)