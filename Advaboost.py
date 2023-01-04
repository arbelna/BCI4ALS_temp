# Load libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from collections import Counter
# from psychopy import visual
import warnings

warnings.filterwarnings('ignore')


class Adva_boost:
    def __init__(self, data_path):
        self.X_Test_happy = None
        self.X_Test_sad = None
        self.test_data = None
        self.clf = None
        self.data_path = data_path
        self.data = self.read_data(self.data_path)
        self.X_Train, self.Y_Train = self.Preprocess(self.data)

    def read_data(self, data_path, train=True):
        if train:
            with open(data_path, 'rb') as f:
                downsampled_target = np.load(f)
                downsampled_non_target = np.load(f)
                downsampled_dist = np.load(f)
                return [downsampled_target, downsampled_non_target, downsampled_dist]
        else:
            with open(data_path, 'rb') as f:
                downsampled_eeg_data_tar0 = np.load(f)
                downsampled_eeg_data_non_tar0 = np.load(f)
                downsampled_eeg_data_dist0 = np.load(f)
                downsampled_eeg_data_tar1 = np.load(f)
                downsampled_eeg_data_non_tar1 = np.load(f)
                downsampled_eeg_data_dist1 = np.load(f)
                return [downsampled_eeg_data_tar0, downsampled_eeg_data_non_tar0, downsampled_eeg_data_dist0]

    def Preprocess(self, data, train=True):
        """
        Transforming the results from the records into input fot the model
        :param train: boolean var to check if to preprocess the date as input for train or test
        :param data
        :return: X, Y
        """
        # [[0, 2, 3, 5, 9]]
        shape_target = data[0][[0, 2, 3, 5, 9]].shape
        result_target = data[0][[0, 2, 3, 5, 9]].reshape(shape_target[0] * shape_target[1], shape_target[2])
        shape_nontarget = data[1][[0, 2, 3, 5, 9]].shape
        result_nontarget = data[1][[0, 2, 3, 5, 9]].reshape(shape_nontarget[0] * shape_nontarget[1], shape_nontarget[2])
        # shape_dist = data[2][[0, 2, 3, 5, 9]].shape
        # result_dist = data[2][[0, 2, 3, 5, 9]].reshape(shape_dist[0] * shape_dist[1], shape_dist[2])[:2500]
        if train:
            # X = np.concatenate((result_target, result_nontarget, result_dist), axis=0)
            X = np.concatenate((result_target, result_nontarget), axis=0)
            # Y = np.concatenate(
            #     [np.ones(shape_target[0] * shape_target[1]), np.zeros(shape_nontarget[0] * shape_nontarget[1]), np.ones(2500)*2])
            Y = np.concatenate(
                [np.ones(shape_target[0] * shape_target[1]), np.zeros(shape_nontarget[0] * shape_nontarget[1])])
            # Y = np.concatenate(
            #     [np.ones(shape_target[1]), np.zeros(shape_nontarget[1])])
            return X, Y
        else:
            return result_target, result_nontarget

    def train_model(self):
        """
        Training the model anf finding the optimal hyperparameters
        :return: The best model ever
        """
        # Split dataset into training set and test set
        # 3/4 training and 1/4 test
        X_train, X_test, y_train, y_test = train_test_split(self.X_Train, self.Y_Train, test_size=0.1,
                                                            random_state=1543)
        # X_train, y_train = self.X_Train, self.Y_Train
        """
        First try with AdaBoost model using SVM as estimator
        
        # accuracy_test_list = []
        # accuracy_train_list = []
        # estimator_errors_list = []
        # n_estims = [1, 3, 10, 50, 100, 1000, 5000]
        # models_list = []
        # for n in n_estims:
        #     svc = LinearSVC(tol=1e-10, loss='hinge', C=100, max_iter=50000)
        #     # svc = SVC(tol=1e-5, degree=50, max_iter=50000)
        #     # Create adaboost classifer object
        #     abc = AdaBoostClassifier(estimator=None, n_estimators=n, algorithm='SAMME')
        #     abc = RandomForestClassifier(n_estimators=n, max_depth=15)
        #     # Train Adaboost Classifer
        #     model = abc.fit(X_train, y_train)
        #     # creating a list that will hold all of the 7 models
        #     models_list.append(model)
        #     # Predict the response for test dataset
        #     y_pred = model.predict(X_test)
        #     y_pred_train = model.predict(X_train)
        #
        #     # Model Accuracy, how often is the classifier correct?
        #     print(f"number of estimators:", n)
        #     print("Test Accuracy:", accuracy_score(y_test, y_pred))
        #     accuracy_test_list.append(accuracy_score(y_test, y_pred))
        #     print("Accuracy on the train data:", accuracy_score(y_train, y_pred_train))
        #     print('')
        #     accuracy_train_list.append(accuracy_score(y_train, y_pred_train))
        #     # print(abc.estimator_errors_ )
        #     # estimator_errors_list.append(abc.estimator_errors_)
        
        # grid-search to RandomForestClassifier in order to find the best hyperparameters
        clf = RandomForestClassifier()
        parameters = {'n_estimators': [1000, 1500, 2000],
                      'criterion': ['gini', 'entropy', 'log_loss'],
                      'max_depth': [15, 20],
                      'max_features': ['sqrt', 'log2', None]
                      }
        warnings.filterwarnings('ignore')

        grid_obj = GridSearchCV(clf, parameters, return_train_score=True, scoring=f1_score, verbose=10)
        grid_obj = grid_obj.fit(X_train, y_train)
        print(grid_obj.best_params_)
        clf = grid_obj.best_estimator_
        clf.fit(X_train, y_train)
        print("Accuracy score on train set:" + str(accuracy_score(y_train, clf.predict(X_train))))
        print("Accuracy score on test set:" + str(accuracy_score(y_test, clf.predict(X_test))))
        cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()
        """

        self.clf = RandomForestClassifier(ccp_alpha=0, criterion='log_loss', max_depth=15, max_features='log2',
                                          n_estimators=1500)
        # self.clf = SVC(tol=1e-5, degree=50, max_iter=50000, probability=True)
        self.clf.fit(X_train, y_train)
        print("Accuracy score on train set:" + str(accuracy_score(y_train, self.clf.predict(X_train))))
        print("Accuracy score on test set:" + str(accuracy_score(y_test, self.clf.predict(X_test))))
        # return accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, clf.predict(X_test))
        cm = confusion_matrix(y_test, self.clf.predict(X_test), labels=self.clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        disp.plot()
        plt.show()

    def test_model(self, path):
        self.test_data = self.read_data(path, False)
        self.X_Test_happy, self.X_Test_sad = self.Preprocess(self.test_data, False)
        # happy_predicted_classes = Counter(self.clf.predict(self.X_Test_happy))
        happy_predicted_classes = self.clf.predict(self.X_Test_happy)
        # happy_predicted_classes = np.array([sum(happy_predicted_classes[0::5]), sum(happy_predicted_classes[1::5]),
        #                                     sum(happy_predicted_classes[2::5]), sum(happy_predicted_classes[3::5]),
        #                                     sum(happy_predicted_classes[4::5])]) / (len(happy_predicted_classes) / 5)
        # happy_predicted_classes = np.add.reduceat(happy_predicted_classes,
        #                                           np.arange(0, len(happy_predicted_classes), 5))
        # happy_predicted_classes = [1 if x > 2.5 else 0 for x in happy_predicted_classes]

        # happy_predicted_classes = [1 if x == 1 else 0 for x in happy_predicted_classes]
        happy_res = 0
        happy_res_per_trail = []
        for index, pred in enumerate(happy_predicted_classes):
            happy_res += pred
            if index % 5 == 4:
                if happy_res > 2.5:
                    happy_res_per_trail.append(1)
                else:
                    happy_res_per_trail.append(0)
                happy_res = 0
        happy_target_chances = sum(happy_res_per_trail) / len(happy_res_per_trail)
        # happy_target_chances = np.average(np.average(happy_predicted_classes, axis=1, weights=[1, 0])) / 5

        # sad_predicted_classes = Counter(self.clf.predict(self.X_Test_sad))
        sad_predicted_classes = self.clf.predict(self.X_Test_sad)
        sad_res = 0
        sad_res_per_trail = []
        for index, pred in enumerate(sad_predicted_classes):
            sad_res += pred
            if index % 5 == 4:
                if sad_res > 2.5:
                    sad_res_per_trail.append(1)
                else:
                    sad_res_per_trail.append(0)
                sad_res = 0
        # sad_predicted_classes = np.array([sum(sad_predicted_classes[0::5]), sum(sad_predicted_classes[1::5]),
        #                                   sum(sad_predicted_classes[2::5]), sum(sad_predicted_classes[3::5]),
        #                                   sum(sad_predicted_classes[4::5])]) / (len(sad_predicted_classes) / 5)
        # sad_predicted_classes = np.add.reduceat(sad_predicted_classes,
        #                                         np.arange(0, len(sad_predicted_classes), 5))
        # sad_predicted_classes = [1 if x > .5 else 0 for x in sad_predicted_classes]

        # sad_predicted_classes = [1 if x == 1 else 0 for x in sad_predicted_classes]
        sad_predicted_chances = sum(sad_res_per_trail) / len(sad_res_per_trail)
        # sad_predicted_chances = np.average(np.average(sad_predicted_classes, axis=1, weights=[1, 0])) / 5
        accuracy = (happy_target_chances * len(happy_res_per_trail) + (1 - sad_predicted_chances)
                    * len(sad_res_per_trail)) / (len(happy_res_per_trail) + len(sad_res_per_trail))
        if happy_target_chances > sad_predicted_chances:
            result = 'Yes'
            # return 'Yes'
        else:
            result = 'no'
            # return 'no'

        # mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")
        # start_block_win = visual.TextStim(mywin, result,
        #                                   color=(1, 1, 1),
        #                                   colorSpace='rgb')
        # start_block_win.draw()
        # mywin.close()

        # print("Accuracy score on test set:" + str(accuracy_score(, self.clf.predict(self.X_Test_happy))))
        # cm = confusion_matrix(, self.clf.predict(self.X_Test_happy), labels=self.clf.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        # disp.plot()
        # plt.show()
