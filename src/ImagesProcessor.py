# Function load_dataset(path_to_folder) - devuelve ndarray de rgb de cada pixel + columna label (1 gato, 0 perro)
# Function get_features_1(imagenes) - retrona dataset de attributes + columna label
# Function get_features_2(imagenes) - retrona dataset de attributes + columna label
# Function get_features_n(imagenes) - retrona dataset de attributes + columna label
import numpy as np
import cv2
import mahotas
import os
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM


class ImagesProcessor:
    FOUR_PIXEL_COMBINATION = [(0, 0, 0, 0),(0, 0, 0, 1),(0, 0, 1, 0),(0, 0, 1, 1),(0, 1, 0, 0),(0, 1, 0, 1),(0, 1, 1, 0),(0, 1, 1, 1),(1, 0, 0, 0),(1, 0, 0, 1),(1, 0, 1, 0),(1, 0, 1, 1),(1, 1, 0, 0),(1, 1, 0, 1),(1, 1, 1, 0),(1, 1, 1, 1)]

    def __init__(self, path, training=False, size=None):
        self.training = training
        self.path = path
        self.images = []
        self.imagesClass = []
        for filename in os.listdir(path):
            if(filename.find('.DS_Store') >= 0): # Filtro los archivos temporales q mete mac en las carpetas
                continue
            pathFileName = path+filename
            image = cv2.imread(pathFileName)
            if size is not None:
                image = cv2.resize(image, size)
            self.loadImage(image, filename)
        self.imagesMahotas = None
        self.grayImages = None
        self.darkPatternFeature = None
        self.flatImages = None
        self.textureFeatures = {} # Contiene todas las texturas calculadas para los distintos valores de radio y puntos
        self.sizes = None
        self.imagesEdges = None

    # Metodo encargado de almacenar la imagen con sus distintas variantes
    def loadImage(self, image, filename):
        self.images.append(image) # Cargo las imagenes en formato cv2
        self.images.append(cv2.flip(image,1)) # Cargo el mirror horizontal
        self.images.append(cv2.flip(image,0)) # Cargo el mirror vertical
        # TODO: Agregar esta transformacion: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#affine-transformation
        if(self.training):
            animalClass = self.getAnimalClass(filename)
            self.imagesClass.append(animalClass) # Al agregar la imagen y su clase en el mismo orden no pierdo la relacion
            self.imagesClass.append(animalClass) # Agrego la clase del mirror Horizontal
            self.imagesClass.append(animalClass) # Agrego la clase del mirror vertical

    # Por ahora no la usamos mas.
    # Carga la imagen en grises con mejor definicion que opencv
    # pero como despues la normalizamos no nos cambia tanto.
    def loadMahotas(self):
        self.imagesMahotas = []
        for filename in os.listdir(self.path):
            pathFileName = self.path+filename
            self.imagesMahotas.append(mahotas.imread(pathFileName, as_grey=True)) # cargo las imagenes en formato mahotas

    def getAnimalClass(self, filename):
        if(filename.find('cat') >= 0):
            return 1
        elif(filename.find('dog') >= 0):
            return 0
        else:
            raise ValueError("El nombre del filename no contiene informacion: %s." % filename)

    def getImages(self):
        return self.images

    def getImagesWithSize(self, size, images=None):
        if images is None:
            images = self.images
        image_array = []
        for img in images:
            image_array += [cv2.resize(img, size)]
        return(image_array)

    def getImagesClass(self):
        return self.imagesClass

    def getImagesSize(self):
        if(self.sizes is None):
            self.sizes = []
            for image in self.images:
                self.sizes.append([len(image[0]), len(image)])
        return self.sizes

    # Retorna el conjunto de las images con el
    # histograma normalizado de imagenes en escala de grises
    def getImagesWithGrayHistogramEqualized(self):
        # De esta forma calculo 1 sola vez el histograma
        if(self.grayImages is None):
            print "Calculando el Histograma Normalizado en Grises"
            self.grayImages = []
            for image in self.images:
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                grayImage = cv2.equalizeHist(grayImage)
                self.grayImages.append(grayImage)
        return self.grayImages

    def oscuro(self, color):
        return 1 if (color <= 127) else 0

    # Alto nombre que le puse ;)
    # Cuanta la cantidad de patrones que hay en la imagen
    # tomando cuadrados de 2x2
    def getDarkPatternFeature(self):
        if(self.darkPatternFeature is None):
            gimages = self.getImagesWithGrayHistogramEqualized()
            self.darkPatternFeature = []
            for image in gimages:
                self.darkPatternFeature.append(self.getDarkPattern(image))
        return self.darkPatternFeature

    def getDarkPattern(self, image):
        patternHits = {}
        for i in range(image.shape[0]-1):
            for j in range(image.shape[1]-1):
                    p = (self.oscuro(image[i, j]), self.oscuro(image[i+1, j]), self.oscuro(image[i+1, j]), self.oscuro(image[i+1, j+1]))
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

    def getTextureFeature(self, radius, points):
        key = (radius, points)
        if(self.textureFeatures.get(key) is None):
            print "Calculando texturas para radio %d con %d puntos" % (radius, points)
            textures = []
            gimages = self.getImagesWithGrayHistogramEqualized()
            for image in gimages:
                textures.append(mahotas.features.lbp(image, radius, points, ignore_zeros=False))
            self.textureFeatures[key] = textures
        return self.textureFeatures[key]

    def getPcaFeatures(self, components, image_size, imageDataset=None):
        if imageDataset is None:
            imageDataset = self.getImagesAsDataset(image_size)
        norm = Normalizer()
        imageDataset = norm.fit_transform(imageDataset)
        pca = PCA(n_components=components)
        imageDataset = pca.fit_transform(imageDataset)
        return pca, norm, imageDataset

    def getImagesAsDataset(self, size, images=None):
        if self.flatImages is None:
            if images is None:
                images = np.array(self.getImagesWithSize(size))
            else:
                images = np.array(self.getImagesWithSize(size, images))
            n = len(images)
            if len(images[0].shape) == 3:
                images = np.array(images).reshape((n, -1, 3)).reshape((n, -1))  # Magia de reshape para obtener n filas con los pixeles de las imagenes aplanados en 1-D
            else:
                images = np.array(images).reshape((n, -1))
            self.flatImages = images
            return(images)
        else:
            return(self.flatImages)

    def getBernulliRBM(self, components, image_size, learning_rate=0.1, n_iter=10):
        imageDataset = self.getImagesAsDataset(image_size)
        imageDataset = (imageDataset - np.min(imageDataset, 0)) / (np.max(imageDataset, 0) + 0.0001)
        rbm = BernoulliRBM(n_components=components, learning_rate=learning_rate, n_iter=n_iter, verbose=True)
        imageDataset = rbm.fit_transform(imageDataset)
        return rbm, imageDataset

    def getImageEdges(self):
        if(self.imagesEdges is None):
            self.imagesEdges = []
            for image in self.images:
                self.imagesEdges.append(cv2.Canny(image,100,200))
        return self.imagesEdges
