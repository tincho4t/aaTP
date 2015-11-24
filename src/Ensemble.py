import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from RandomForest import RandomForest
from RidgeClassifier import RidgeClassifier
from ImagesProcessor import ImagesProcessor
from Constants import Constants
import threading
import time

class Ensemble(object):

    def __init__(self):
        self.pca_randomForest = None
        self.pca_randomForest_norm = None
        self.pca_randomForest_pca = None
        self.rbm_lr_rbm = None
        self.rbm_lr = None
        self.texture_10_8 = None
        self.texture_5_10 = None
        self.texture_7_10 = None
        self.texture_9_8 = None
        self.texture_4_10 = None
        self.texture_20_8 = None
        self.ensemble_logistic_regression = None
        self.edge_pca_lr = None
        self.pca_edge_norm = None
        self.pca_edge_pca = None
        self.ip = ImagesProcessor()
        # Agregamos las predicciones aca porque no logramos pasarlas por referencia
        self.pca_randomForest_y_hat = None
        self.rbm_lr_y_hat = None
        self.texture_10_8_y_hat = None
        self.texture_5_10_y_hat = None

    def fit_small(self, images, y):
        images_transformed, y_transformed = self.ip.transformImages(images, y, rotate=True, crop=True)
        
        t_t10_8 = threading.Thread(target=self._fit_small_texture10_8, args=(images[:], y, self.texture_10_8, 10, 8, 2))
        t_t10_8.daemon = True
        t_t10_8.start()

        t_t5_10 = threading.Thread(target=self._fit_small_texture5_10, args=(images[:], y, self.texture_5_10, 5, 10, 2))
        t_t5_10.daemon = True
        t_t5_10.start()

        t_t7_10 = threading.Thread(target=self._fit_small_texture7_10, args=(images[:], y, self.texture_7_10, 7, 10, 2))
        t_t7_10.daemon = True
        t_t7_10.start()

        t_t9_8 = threading.Thread(target=self._fit_small_texture9_8, args=(images[:], y, self.texture_9_8, 9, 8, 2))
        t_t9_8.daemon = True
        t_t9_8.start()

        t_t4_10 = threading.Thread(target=self._fit_small_texture4_10, args=(images[:], y, self.texture_4_10, 4, 10, 2))
        t_t4_10.daemon = True
        t_t4_10.start()

        t_t20_8 = threading.Thread(target=self._fit_small_texture20_8, args=(images[:], y, self.texture_20_8, 20, 8, 2))
        t_t20_8.daemon = True
        t_t20_8.start()

        t_pc = threading.Thread(target=self._fit_small_pc, args=(images_transformed[:], y_transformed))
        t_pc.daemon = True
        t_pc.start()

        t_rbm = threading.Thread(target=self._fit_small_rbm, args=(images_transformed[:], y_transformed))
        t_rbm.daemon = True
        t_rbm.start()

        t_t10_8.join()
        t_t5_10.join()
        t_t7_10.join()
        t_t9_8.join()
        t_t4_10.join()
        t_t20_8.join()
        t_pc.join()
        t_rbm.join()
        

    def _fit_small_texture10_8(self, images, y, estimator, radius, points, alpha):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_10_8 = RidgeClassifier(ds, y, alpha=alpha)
        self.texture_10_8.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    # FIXE: unificar estas dos funciones. No le gusta pasar el estimador como atributo
    def _fit_small_texture5_10(self, images, y, estimator, radius, points, alpha):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_5_10 = RidgeClassifier(ds, y, alpha=alpha)
        self.texture_5_10.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    def _fit_small_texture7_10(self, images, y, estimator, radius, points, alpha):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_7_10 = RidgeClassifier(ds, y, alpha=alpha)
        self.texture_7_10.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    def _fit_small_texture9_8(self, images, y, estimator, radius, points, alpha):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_9_8 = RidgeClassifier(ds, y, alpha=alpha)
        self.texture_9_8.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    def _fit_small_texture4_10(self, images, y, estimator, radius, points, alpha):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_4_10 = RidgeClassifier(ds, y, alpha=alpha)
        self.texture_4_10.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    def _fit_small_texture20_8(self, images, y, estimator, radius, points, alpha):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_20_8 = RidgeClassifier(ds, y, alpha=alpha)
        self.texture_20_8.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    def _fit_small_pc(self, images, y):
        start_time = time.time()
        print("PCA RANDOM FOREST")
        ds = self.ip.getImagesWithGrayHistogramEqualized(images=images)
        self.pca_randomForest_pca, self.pca_randomForest_norm, ds = self.ip.getPcaFeatures(ds, 150, Constants.IMAGES_SIZES)
        self.pca_randomForest = RandomForest(ds, y, n_estimators=2000)
        self.pca_randomForest.fit()
        print("COMPELTE PCA RANDOM FOREST --- %s seconds ---" %(time.time() - start_time))

    def _fit_small_rbm(self, ds, y):
        start_time = time.time()
        print("RBM LR")
        ds = self.ip.getImagesAsDataset(ds, Constants.IMAGES_SIZES)
        ds = (ds - np.min(ds, 0)) / (np.max(ds, 0) + 0.0001)
        self.rbm_lr_rbm = BernoulliRBM(random_state=0, verbose=True)
        self.rbm_lr_rbm.learning_rate = 0.01
        self.rbm_lr_rbm.n_iter = 5
        self.rbm_lr_rbm.n_components = 150
        logistic = linear_model.RidgeClassifier(alpha=2)
        self.rbm_lr = Pipeline(steps=[('rbm', self.rbm_lr_rbm), ('lr', logistic)])
        self.rbm_lr.fit(ds, y)
        print("COMPLETE RBM LR --- %s seconds ---" % (time.time() - start_time))


    def fit_big(self, ds, y):
        self.ensemble_logistic_regression = linear_model.LogisticRegression()
        self.ensemble_logistic_regression.fit(ds, y)

    def predict_small(self, images):

        t_predict_small_pac_ranfomForest = threading.Thread(target=self._predict_small_pac_ranfomForest, args=(images, ))
        t_predict_small_pac_ranfomForest.daemon = True
        t_predict_small_pac_ranfomForest.start()

        t_predict_small_rbm_lr = threading.Thread(target=self._predict_small_rbm_lr, args=(images, ))
        t_predict_small_rbm_lr.daemon = True
        t_predict_small_rbm_lr.start()

        t_predict_small_texture_10_8 = threading.Thread(target=self._predict_small_texture_10_8, args=(images, ))
        t_predict_small_texture_10_8.daemon = True
        t_predict_small_texture_10_8.start()

        t_predict_small_texture_5_10 = threading.Thread(target=self._predict_small_texture_5_10, args=(images, ))
        t_predict_small_texture_5_10.daemon = True
        t_predict_small_texture_5_10.start()

        t_predict_small_texture_7_10 = threading.Thread(target=self._predict_small_texture_7_10, args=(images, ))
        t_predict_small_texture_7_10.daemon = True
        t_predict_small_texture_7_10.start()

        t_predict_small_texture_9_8 = threading.Thread(target=self._predict_small_texture_9_8, args=(images, ))
        t_predict_small_texture_9_8.daemon = True
        t_predict_small_texture_9_8.start()

        t_predict_small_texture_4_10 = threading.Thread(target=self._predict_small_texture_4_10, args=(images, ))
        t_predict_small_texture_4_10.daemon = True
        t_predict_small_texture_4_10.start()

        t_predict_small_texture_20_8 = threading.Thread(target=self._predict_small_texture_20_8, args=(images, ))
        t_predict_small_texture_20_8.daemon = True
        t_predict_small_texture_20_8.start()

        t_predict_small_pac_ranfomForest.join()
        t_predict_small_rbm_lr.join()
        t_predict_small_texture_10_8.join()
        t_predict_small_texture_5_10.join()
        t_predict_small_texture_9_8.join()
        t_predict_small_texture_4_10.join()
        t_predict_small_texture_20_8.join()

        return(np.vstack((self.pca_randomForest_y_hat, self.rbm_lr_y_hat, self.texture_10_8_y_hat, self.texture_5_10_y_hat)).T)


    def _predict_small_rbm_lr(self, images):
        start_time = time.time()
        ds = images[:]
        ds = self.ip.getImagesAsDataset(ds, Constants.IMAGES_SIZES)
        ds = (ds - np.min(ds, 0)) / (np.max(ds, 0) + 0.0001)
        self.rbm_lr_y_hat = self.rbm_lr.predict(ds)
        print "Complete prediction RBM --- %s ---" % (time.time() - start_time)

    def _predict_small_pac_ranfomForest(self, images):
        start_time = time.time()
        ds = self.ip.getImagesWithGrayHistogramEqualized(images=images)
        ds = self.ip.getImagesAsDataset(ds, Constants.IMAGES_SIZES)
        ds = self.pca_randomForest_norm.transform(ds)
        ds = self.pca_randomForest_pca.transform(ds)
        self.pca_randomForest_y_hat = self.pca_randomForest.predict(ds)
        print "Complete prediction PCA --- %s ---" % (time.time() - start_time)

    def _predict_small_texture_10_8(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 10, 8)
        self.texture_10_8_y_hat = self.texture_10_8.predict(ds)
        print "Complete prediction Texture 10 8 --- %s ---" % (time.time() - start_time)

    def _predict_small_texture_5_10(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 5, 10)
        self.texture_5_10_y_hat = self.texture_5_10.predict(ds)
        print "Complete prediction Texture 5 10 --- %s ---" % (time.time() - start_time)
    
    def _predict_small_texture_7_10(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 7, 10)
        self.texture_7_10_y_hat = self.texture_7_10.predict(ds)
        print "Complete prediction Texture 7 10 --- %s ---" % (time.time() - start_time)
    
    def _predict_small_texture_9_8(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 9, 8)
        self.texture_9_8_y_hat = self.texture_9_8.predict(ds)
        print "Complete prediction Texture 9 8 --- %s ---" % (time.time() - start_time)

    def _predict_small_texture_4_10(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 4, 10)
        self.texture_4_10_y_hat = self.texture_4_10.predict(ds)
        print "Complete prediction Texture 4 10 --- %s ---" % (time.time() - start_time)
    
    def _predict_small_texture_20_8(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 20, 8)
        self.texture_20_8_y_hat = self.texture_20_8.predict(ds)
        print "Complete prediction Texture 20 8 --- %s ---" % (time.time() - start_time)
    
    def predict_big(self, ds):
        return(self.ensemble_logistic_regression.predict(ds))
