import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from RandomForest import RandomForest
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
        self.texture_10_10_randomForest = None
        self.texture_5_10_randomForest = None
        self.ensemble_logistic_regression = None
        self.edge_pca_lr = None
        self.pca_edge_norm = None
        self.pca_edge_pca = None
        self.ip = ImagesProcessor()
        # Agregamos las predicciones aca porque no logramos pasarlas por referencia
        self.pca_randomForest_y_hat = None
        self.rbm_lr_y_hat = None
        self.texture_10_10_randomForest_y_hat = None
        self.texture_5_10_randomForest_y_hat = None

    def fit_small(self, images, y):
        images_transformed, y_transformed = self.ip.transformImages(images, y, rotate=True, crop=True)
        t_pc = threading.Thread(target=self._fit_small_pc, args=(images_transformed[:], y_transformed))
        t_pc.daemon = True
        t_pc.start()

        t_rbm = threading.Thread(target=self._fit_small_rbm, args=(images_transformed[:], y_transformed))
        t_rbm.daemon = True
        t_rbm.start()

        t_t10_10 = threading.Thread(target=self._fit_small_texture1, args=(images[:], y, self.texture_10_10_randomForest, 5, 10, 2000))
        t_t10_10.daemon = True
        t_t10_10.start()

        t_t5_10 = threading.Thread(target=self._fit_small_texture2, args=(images[:], y, self.texture_5_10_randomForest, 5, 10, 2000))
        t_t5_10.daemon = True
        t_t5_10.start()

        t_pc.join()
        t_rbm.join()
        t_t10_10.join()
        t_t5_10.join()

        # self._fit_small_pc(images[:], y)
        # self._fit_small_rbm(images[:], y)
        # self._fit_small_texture1(images[:], y, self.texture_5_10_randomForest, 5, 10, 2000)
        # self._fit_small_texture2(images[:], y, self.texture_10_10_randomForest, 10, 10, 2000)

    # FIXE: unificar estas dos funciones. No le gusta pasar el estimador como atributo
    def _fit_small_texture1(self, images, y, estimator, radius, points, n_estimators):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_5_10_randomForest = RandomForest(ds, y, n_estimators=n_estimators)
        self.texture_5_10_randomForest.fit()
        print("COMPLETE TEXTURE %d %d --- %s seconds ---" % (radius, points, time.time() - start_time))

    def _fit_small_texture2(self, images, y, estimator, radius, points, n_estimators):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_10_10_randomForest = RandomForest(ds, y, n_estimators=n_estimators)
        self.texture_10_10_randomForest.fit()
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

        t_predict_small_texture_10_10_randomForest = threading.Thread(target=self._predict_small_texture_10_10_randomForest, args=(images, ))
        t_predict_small_texture_10_10_randomForest.daemon = True
        t_predict_small_texture_10_10_randomForest.start()

        t_predict_small_texture_5_10_randomForest = threading.Thread(target=self._predict_small_texture_5_10_randomForest, args=(images, ))
        t_predict_small_texture_5_10_randomForest.daemon = True
        t_predict_small_texture_5_10_randomForest.start()

        t_predict_small_pac_ranfomForest.join()
        t_predict_small_rbm_lr.join()
        t_predict_small_texture_10_10_randomForest.join()
        t_predict_small_texture_5_10_randomForest.join()

        return(np.vstack((self.pca_randomForest_y_hat, self.rbm_lr_y_hat, self.texture_10_10_randomForest_y_hat, self.texture_5_10_randomForest_y_hat)).T)


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

    def _predict_small_texture_10_10_randomForest(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 10, 10)
        self.texture_10_10_randomForest_y_hat = self.texture_10_10_randomForest.predict(ds)
        print "Complete prediction Texture 10 10 --- %s ---" % (time.time() - start_time)

    def _predict_small_texture_5_10_randomForest(self, images):
        start_time = time.time()
        ds = self.ip.getTextureFeature(images, 5, 10)
        self.texture_5_10_randomForest_y_hat = self.texture_5_10_randomForest.predict(ds)
        print "Complete prediction Texture 5 10 --- %s ---" % (time.time() - start_time)


    def predict_big(self, ds):
        return(self.ensemble_logistic_regression.predict(ds))
