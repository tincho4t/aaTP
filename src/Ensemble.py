import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from RandomForest import RandomForest
from ImagesProcessor import ImagesProcessor


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
        self.ip = ImagesProcessor()

    def fit_small(self, images, y):
        print("PCA RANDOM FOREST")
        ds = self.ip.getImagesWithGrayHistogramEqualized(images=images)
        self.pca_randomForest_pca, self.pca_randomForest_norm, ds = self.ip.getPcaFeatures(ds, 20, (56, 56))
        self.pca_randomForest = RandomForest(ds, y, n_estimators=2000)
        self.pca_randomForest.fit()

        print("RBM LR")
        ds = images[:]
        ds = self.ip.getImagesAsDataset(ds, (56, 56))
        ds = (ds - np.min(ds, 0)) / (np.max(ds, 0) + 0.0001)
        self.rbm_lr_rbm = BernoulliRBM(random_state=0, verbose=True)
        self.rbm_lr_rbm.learning_rate = 0.005
        self.rbm_lr_rbm.n_iter = 20
        self.rbm_lr_rbm.n_components = 100
        logistic = linear_model.LogisticRegression()
        self.rbm_lr = Pipeline(steps=[('rbm', self.rbm_lr_rbm), ('lr', logistic)])
        self.rbm_lr.fit(ds, y)

        print("TEXTURE 10 10")
        ds = self.ip.getTextureFeature(images, 10, 10)
        self.texture_10_10_randomForest = RandomForest(ds, y, n_estimators=2000)
        self.texture_10_10_randomForest.fit()

        print("TEXTURE 5 10")
        ds = self.ip.getTextureFeature(images, 5, 10)
        self.texture_5_10_randomForest = RandomForest(ds, y, n_estimators=2000)
        self.texture_5_10_randomForest.fit()

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
