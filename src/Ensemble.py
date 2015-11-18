import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from RandomForest import RandomForest
from ImagesProcessor import ImagesProcessor
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
        print("COMPLETE TEXTURE %d %d" % (radius, points))
        print("--- %s seconds ---" % (time.time() - start_time))

    def _fit_small_texture2(self, images, y, estimator, radius, points, n_estimators):
        start_time = time.time()
        print("TEXTURE %d %d" % (radius, points))
        ds = self.ip.getTextureFeature(images, radius, points)
        self.texture_10_10_randomForest = RandomForest(ds, y, n_estimators=n_estimators)
        self.texture_10_10_randomForest.fit()
        print("COMPLETE TEXTURE %d %d" % (radius, points))
        print("--- %s seconds ---" % (time.time() - start_time))

    def _fit_small_pc(self, images, y):
        start_time = time.time()
        print("PCA RANDOM FOREST")
        ds = self.ip.getImagesWithGrayHistogramEqualized(images=images)
        self.pca_randomForest_pca, self.pca_randomForest_norm, ds = self.ip.getPcaFeatures(ds, 150, (56, 56))
        self.pca_randomForest = RandomForest(ds, y, n_estimators=2000)
        self.pca_randomForest.fit()
        print("COMPELTE PCA RANDOM FOREST")
        print("--- %s seconds ---" % (time.time() - start_time))

    def _fit_small_rbm(self, ds, y):
        start_time = time.time()
        print("RBM LR")
        ds = self.ip.getImagesAsDataset(ds, (56, 56))
        ds = (ds - np.min(ds, 0)) / (np.max(ds, 0) + 0.0001)
        self.rbm_lr_rbm = BernoulliRBM(random_state=0, verbose=True)
        self.rbm_lr_rbm.learning_rate = 0.01
        self.rbm_lr_rbm.n_iter = 5
        self.rbm_lr_rbm.n_components = 150
        logistic = linear_model.RidgeClassifier(alpha=2)
        self.rbm_lr = Pipeline(steps=[('rbm', self.rbm_lr_rbm), ('lr', logistic)])
        self.rbm_lr.fit(ds, y)
        print("COMPLETE RBM LR")
        print("--- %s seconds ---" % (time.time() - start_time))

    def fit_big(self, ds, y):
        self.ensemble_logistic_regression = linear_model.LogisticRegression()
        self.ensemble_logistic_regression.fit(ds, y)

    def predict_small(self, images):
        ds = self.ip.getImagesWithGrayHistogramEqualized(images=images)
        ds = self.ip.getImagesAsDataset(ds, (56, 56))
        ds = self.pca_randomForest_norm.transform(ds)
        ds = self.pca_randomForest_pca.transform(ds)
        pca_randomForest_y_hat = self.pca_randomForest.predict(ds)

        ds = images[:]
        ds = self.ip.getImagesAsDataset(ds, (56, 56))
        ds = (ds - np.min(ds, 0)) / (np.max(ds, 0) + 0.0001)
        rbm_lr_y_hat = self.rbm_lr.predict(ds)

        ds = self.ip.getTextureFeature(images, 10, 10)
        texture_10_10_randomForest_y_hat = self.texture_10_10_randomForest.predict(ds)

        ds = self.ip.getTextureFeature(images, 5, 10)
        texture_5_10_randomForest_y_hat = self.texture_5_10_randomForest.predict(ds)

        return(np.vstack((pca_randomForest_y_hat, rbm_lr_y_hat, texture_10_10_randomForest_y_hat, texture_5_10_randomForest_y_hat)).T)

    def predict_big(self, ds):
        return(self.ensemble_logistic_regression.predict(ds))
