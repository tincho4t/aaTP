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

def oscuro(color):
    return 1 if (color <= 127) else 0


FOUR_PIXEL_COMBINATION = [(0, 0, 0, 0),(0, 0, 0, 1),(0, 0, 1, 0),(0, 0, 1, 1),(0, 1, 0, 0),(0, 1, 0, 1),(0, 1, 1, 0),(0, 1, 1, 1),(1, 0, 0, 0),(1, 0, 0, 1),(1, 0, 1, 0),(1, 0, 1, 1),(1, 1, 0, 0),(1, 1, 0, 1),(1, 1, 1, 0),(1, 1, 1, 1)]

patternHits = {}
for i in range(image.shape[0]-1):
    for j in range(image.shape[1]-1):
            p = (oscuro(image[i,j]), oscuro(image[i+1,j]), oscuro(image[i+1,j]), oscuro(image[i+1,j+1]))
            if p in patternHits.keys():
                patternHits[p] = patternHits[p] + 1
            else:
                patternHits[p] = 1

pixelCombination = []

for combination in FOUR_PIXEL_COMBINATION:
    hits = patternHits.get(combination)
    hits = 0 if hits is None else hits
    pixelCombination.append(hits) # Al agreguar los elementos en el orden de PIXEL_COMBINATION me quedan siempre en orden
