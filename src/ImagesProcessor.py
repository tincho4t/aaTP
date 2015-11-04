# Function load_dataset(path_to_folder) - devuelve ndarray de rgb de cada pixel + columna label (1 gato, 0 perro)
# Function get_features_1(imagenes) - retrona dataset de attributes + columna label
# Function get_features_2(imagenes) - retrona dataset de attributes + columna label
# Function get_features_n(imagenes) - retrona dataset de attributes + columna label
import numpy as np
import cv2
import os

class ImagesProcessor:
    def __init__(self, path, training = False):
        self.training = training
        self.path = path
        self.images = []
        for filename in os.listdir(path):
            pathFileName = path+filename
            animalClass = self.getAnimalClass(filename)
            self.images.append([cv2.imread(pathFileName), animalClass])
        self.grayImages = None


    def getAnimalClass(self, filename):
        if(not self.training):
            return -1 # Devuelvo "Fruta" para mantener la estructura de de datos de images
        elif(filename.find('cat') >= 0):
            return 1
        elif(filename.find('dog') >= 0):
            return 0
        else:
            raise ValueError("El nombre del filename no contiene informacion: %s." % filename)


    def getImages(self):
        return self.images


    # Retorna el conjunto de las images con el
    # histograma normalizado de imagenes en escala de grises
    def getImagesWithGrayHistogramEqualized(self):
        # De esta forma calculo 1 sola vez el histograma
        if(self.grayImages is None):
            print "Calculando el Histograma Normalizado en Grises"
            self.grayImages = []
            for image in self.images:
                grayImage = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
                grayImage = cv2.equalizeHist(grayImage)
                self.grayImages.append([grayImage, image[1]]) # Imagen y su clasificacion
        return self.grayImages


    def oscuro(color):
        return (color <= 127)? 1 : 0

"""
    # Alto nombre que le puse ;)
    # Cuanta la cantidad de patrones que hay en la imagen
    # tomando cuadrados de 2x2
    def getDarkPatterns(self, image):
        pattern_dict = {}
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                    p = (self.oscuro(image[i,j]), self.oscuro(image[i+1,j]), self.oscuro(image[i+1,j]), self.oscuro(image[i+1,j+1]))
                    if p in pattern_dict.keys():
                        pattern_dict[p] = pattern_dict[p] + 1
                    else:
                        pattern_dict[p] = 1
        return pattern_dict
"""

