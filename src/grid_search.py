from Constants import Constants
from ImagesProcessor import ImagesProcessor

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import rlcompleter, readline
readline.parse_and_bind('tab:complete')

from sklearn.ensemble import RandomForestClassifier

# Setear Dataset
ip = ImagesProcessor('../imgs/test/medium/', training=True)

# Setear Features
X = ip.getTextureFeature(5, 12)

# Setear Parametros para tunear
tuned_parameters = [{'n_estimators': [500, 850, 1000, 1500, 2000, 5000]}]

# Setear Clasificador para tuner
classifier = RandomForestClassifier()

#############################################################
######## DE ACA PARA ABAJO NO HACE FALTA TOCAR NADA #########
#############################################################

scores = ['precision', 'recall']
Y = ip.getImagesClass()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=Constants.TEST_SIZE, random_state=Constants.RANDOM_STATE)
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(classifier, tuned_parameters, cv=10, scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
