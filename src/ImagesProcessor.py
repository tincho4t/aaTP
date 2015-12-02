import numpy as np
import cv2
import mahotas
import os
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
import random


class ImagesProcessor:
    FOUR_PIXEL_COMBINATION = [(0, 0, 0, 0),(0, 0, 0, 1),(0, 0, 1, 0),(0, 0, 1, 1),(0, 1, 0, 0),(0, 1, 0, 1),(0, 1, 1, 0),(0, 1, 1, 1),(1, 0, 0, 0),(1, 0, 0, 1),(1, 0, 1, 0),(1, 0, 1, 1),(1, 1, 0, 0),(1, 1, 0, 1),(1, 1, 1, 0),(1, 1, 1, 1)]

    # Levanta las imagenes del directorio.
    # Devuelve una tupla (imagenes, clase de las images)
    # Si trainig es false, classes va a ser []
    # Si sizes != None entonces hace un resize de las images
    def getImages(self, path, size=None, training=False):
        images = []
        classes = []
        for filename in os.listdir(path):
            if(filename.find('.DS_Store') >= 0): # Filtro los archivos temporales q mete mac en las carpetas
                continue
            pathFileName = path+filename
            image = cv2.imread(pathFileName)
            if size is not None:
                image = cv2.resize(image, size)
            if(training):
                classes.append(self._getAnimalClass(filename))
            else:
                classes.append(filename)
            images.append(image)
        return (images, classes)

    def _getAnimalClass(self, filename):
        if(filename.find('cat') >= 0):
            return 1
        elif(filename.find('dog') >= 0):
            return 0
        else:
            raise ValueError("El nombre del filename no contiene informacion: %s." % filename)

    def _rotateImage(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return(rotated)

    # Le entregas las imagenes y sus clases y te devuelve
    # una tupla con (images originales mas las transformadas, sus respectivas clases)
    def transformImages(self, images, classes=None, rotate=False, crop=False):
        transformedImages = []
        transformedClasses = []
        for i in range(0, len(images)):
            image = images[i]
            transformedImages.append(image) # Cargo las imagenes en formato cv2
            transformedImages.append(cv2.flip(image, 1)) # Cargo el mirror horizontal
            transformedImages.append(cv2.flip(image, 0)) # Cargo el mirror vertical
            if rotate:
                transformedImages.append(self._rotateImage(image, 90))
                transformedImages.append(self._rotateImage(image, 270))
            if crop:
                transformedImages += self._getCrops(image)
            # TODO: Agregar esta transformacion: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#affine-transformation
            if(classes is not None):
                animalClass = classes[i]
                transformedClasses.append(animalClass) # Al agregar la imagen y su clase en el mismo orden no pierdo la relacion
                transformedClasses.append(animalClass) # Agrego la clase del mirror Horizontal
                transformedClasses.append(animalClass) # Agrego la clase del mirror vertical
                if rotate:
                    transformedClasses.append(animalClass)
                    transformedClasses.append(animalClass)
                if crop:
                    for j in range(6):
                        transformedClasses.append(animalClass)
        return (transformedImages, transformedClasses)

    # Devuelve una lista con los tamanios de las imagenes
    def getImagesSize(self, images):
        sizes = []
        for image in images:
            sizes.append([len(image[0]), len(image)])
        return sizes

    # Retorna la misma lista de imagenes pero en escala de grises
    # y con su histograma normalizado de imagenes en escala de grises
    def getImagesWithGrayHistogramEqualized(self, images):
        grayImages = []
        for image in images:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayImage = cv2.equalizeHist(grayImage)
            grayImages.append(grayImage)
        return grayImages

    def _oscuro(self, color):
        return 1 if (color <= 127) else 0

    # Alto nombre que le puse ;)
    # Cuanta la cantidad de patrones que hay en la imagen
    # tomando cuadrados de 2x2
    def getDarkPatternFeature(self, images, classes=None):
        gimages = self.getImagesWithGrayHistogramEqualized(images)
        darkPatternFeature = []
        for image in gimages:
            darkPatternFeature.append(self._getDarkPattern(image))
        return darkPatternFeature

    def _getDarkPattern(self, image):
        patternHits = {}
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                    p = (self._oscuro(image[i, j]), self._oscuro(image[i+1, j]), self._oscuro(image[i+1, j]), self._oscuro(image[i+1, j+1]))
                    if p in patternHits.keys():
                        patternHits[p] = patternHits[p] + 1
                    else:
                        patternHits[p] = 1
        pixelCombination = []
        for combination in self.FOUR_PIXEL_COMBINATION:
            hits = patternHits.get(combination)
            hits = 0 if hits is None else hits
            pixelCombination.append(hits)  # Al agreguar los elementos en el orden de PIXEL_COMBINATION me quedan siempre en orden
        return pixelCombination

    def getTextureFeature(self, images, radius, points):
        textures = []
        gimages = self.getImagesWithGrayHistogramEqualized(images)
        for image in gimages:
            textures.append(mahotas.features.lbp(image, radius, points, ignore_zeros=False))
        return textures

    # Recive imagenes de tamanio variable. Internamente nos encargamos de normalizarlas
    def getPcaFeatures(self, images, components, image_size):
        imageDataset = self.getImagesAsDataset(images, image_size)
        norm = Normalizer()
        imageDataset = norm.fit_transform(imageDataset)
        pca = PCA(n_components=components)
        imageDataset = pca.fit_transform(imageDataset)
        return pca, norm, imageDataset

    # Aplana las imagenes a una dimension
    def getImagesAsDataset(self, images, size):
        images = np.array(self.getImagesWithSize(images, size))
        n = len(images)
        if len(images[0].shape) == 3:
            images = np.array(images).reshape((n, -1, 3)).reshape((n, -1))  # Magia de reshape para obtener n filas con los pixeles de las imagenes aplanados en 1-D
        else:
            images = np.array(images).reshape((n, -1))
        return images

    def getImagesWithSize(self, images, size):
        image_array = []
        for img in images:
            image_array += [cv2.resize(img, size)]
        return image_array

    def getImageEdges(self, images):
        imagesEdges = []
        for image in images:
            imagesEdges.append(cv2.Canny(image, 100, 200))
        return imagesEdges

    def _getCrops(self, image):
        size = image.shape
        imagesCropes = []
        x_len = size[0]
        y_len = size[1]
        imagesCropes.append(image[0:int(x_len*0.7), :])
        imagesCropes.append(image[0:int(x_len*0.7), 0:int(y_len*0.7)])
        imagesCropes.append(image[:, 0:int(y_len*0.7)])
        imagesCropes.append(image[int(x_len*0.3):, :])
        imagesCropes.append(image[:, int(y_len*0.3):])
        imagesCropes.append(image[int(x_len*0.3):, int(y_len*0.3):])
        imagesCropes = self.getImagesWithSize(imagesCropes, (x_len, y_len))
        return imagesCropes
