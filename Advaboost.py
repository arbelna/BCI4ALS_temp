import numpy as np
import pandas as pd
# Load libraries
from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics import make_scorer, accuracy_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
# Import train_test_split function
from sklearn.model_selection import train_test_split

class Adva_boost:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.read_data(self.data_path)
        self.X, self.Y = self.Preprocess(self.data)

    def read_data(self, data_path):
        #
        with open(data_path, 'rb') as f:
            downsampled_target = np.load(f)
            downsampled_non_target = np.load(f)
            downsampled_dist = np.load(f)
            return [downsampled_target, downsampled_non_target, downsampled_dist]

    def Preprocess(self, data):
        shape_target = data[0][[0,1,2,3]].shape
        result_target = data[0][[0,1,2,3]].reshape(shape_target[0]*shape_target[1], shape_target[2])
        shape_nontarget = data[1][[0,1,2,3]].shape
        result_nontarget = data[1][[0,1,2,3]].reshape(shape_nontarget[0] * shape_nontarget[1], shape_nontarget[2])
        X = np.concatenate((result_target, result_nontarget), axis=0)
        Y = np.concatenate([np.ones(shape_target[0]*shape_target[1]), np.zeros(shape_nontarget[0]*shape_nontarget[1])])
        return X, Y

    def train_model(self):
        # Split dataset into training set and test set
        # 3/4 training and 1/4 test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=1 / 4, random_state=1)

        accuracy_test_list = []
        accuracy_train_list = []
        estimator_errors_list = []
        n_estims = [1, 3, 10, 50, 100, 1000, 5000]
        models_list = []
        for n in n_estims:
            svc = LinearSVC(tol=1e-10, loss='hinge', C=100, max_iter=50000)
            # svc = SVC(tol=1e-5, degree=50, max_iter=50000)
            # Create adaboost classifer object
            abc = AdaBoostClassifier(estimator=None, n_estimators=n, algorithm='SAMME')
            abc = RandomForestClassifier(n_estimators=n, max_depth=15)
            # Train Adaboost Classifer
            model = abc.fit(X_train, y_train)
            # creating a list that will hold all of the 7 models
            models_list.append(model)
            # Predict the response for test dataset
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            # Model Accuracy, how often is the classifier correct?
            print(f"number of estimators:", n)
            print("Test Accuracy:", accuracy_score(y_test, y_pred))
            accuracy_test_list.append(accuracy_score(y_test, y_pred))
            print("Accuracy on the train data:", accuracy_score(y_train, y_pred_train))
            print('')
            accuracy_train_list.append(accuracy_score(y_train, y_pred_train))
            # print(abc.estimator_errors_ )
            # estimator_errors_list.append(abc.estimator_errors_)
