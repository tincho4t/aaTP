# La idea de este archivo es tener un script pequenio que nos permita
# probar las distintas partes del codigo

import numpy as np
import cv2
import os
from ImagesProcessor import ImagesProcessor

ip = ImagesProcessor('../imgs/test/', training=True)

images = ip.getImages()
gimages = ip.getImagesWithGrayHistogramEqualized()

## Como mostrar una foto
lala = cv2.imread('../imgs/test/cat.0.jpg')
cv2.imshow('color_image',lala)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows

def oscuro(color):
    return (color <= 127)? 1 : 0

patters_dict = {}
for i in range(image.shape[0]-1):
    for j in range(image.shape[1]-1):
            p = (oscuro(image[i,j]), oscuro(image[i+1,j]), oscuro(image[i+1,j]), oscuro(image[i+1,j+1]))
            if p in patters_dict.keys():
                patters_dict[p] = patters_dict[p] + 1
            else:
                patters_dict[p] = 1
