# Load libraries
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from psychopy import visual
import warnings
warnings.filterwarnings('ignore')


class model:
    def __init__(self, data_path):
        self.X_Test_happy = None
        self.X_Test_sad = None
        self.test_data = None
        self.clf = None
        self.data_path = data_path
        self.data = self.read_data(self.data_path)
        self.X_Train, self.Y_Train = self.Preprocess(self.data)

    def read_data(self, data_path):
        with open(data_path, 'rb') as f:
            downsampled_target = np.load(f)
            downsampled_non_target = np.load(f)
            downsampled_dist = np.load(f)
            return [downsampled_target, downsampled_non_target, downsampled_dist]

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
        if train:
            X = np.concatenate((result_target, result_nontarget), axis=0)
            Y = np.concatenate(
                [np.ones(shape_target[0] * shape_target[1]), np.zeros(shape_nontarget[0] * shape_nontarget[1])])
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
                                                            random_state=1312)

        self.clf = RandomForestClassifier(ccp_alpha=0, criterion='log_loss', max_depth=15, max_features='log2',
                                          n_estimators=1500)
        self.clf.fit(X_train, y_train)
        print("Accuracy score on train set:" + str(accuracy_score(y_train, self.clf.predict(X_train))))
        print("Accuracy score on test set:" + str(accuracy_score(y_test, self.clf.predict(X_test))))
        cm = confusion_matrix(y_train, self.clf.predict(X_train), labels=self.clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        disp.plot()
        plt.show()
        cm = confusion_matrix(y_test, self.clf.predict(X_test), labels=self.clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        disp.plot()
        plt.show()

    def test_model(self, happy, sad):
        self.X_Test_happy, self.X_Test_sad = np.concatenate(happy[[0, 2, 3, 5, 9]]), \
                                             np.concatenate(sad[[0, 2, 3, 5, 9]])
        # predict for each channel in each trail
        happy_predicted_classes = self.clf.predict(self.X_Test_happy)
        happy_res = 0
        happy_res_per_trail = []
        # creating a list with prediction for each trail
        for index, pred in enumerate(happy_predicted_classes):
            happy_res += pred
            if index % 5 == 4:
                if happy_res > 2.5:
                    happy_res_per_trail.append(1)
                else:
                    happy_res_per_trail.append(0)
                happy_res = 0
        happy_target_chances = sum(happy_res_per_trail) / len(happy_res_per_trail)

        # predict for each channel in each trail
        sad_predicted_classes = self.clf.predict(self.X_Test_sad)
        sad_res = 0
        sad_res_per_trail = []
        # creating a list with prediction for each trail
        for index, pred in enumerate(sad_predicted_classes):
            sad_res += pred
            if index % 5 == 4:
                if sad_res > 2.5:
                    sad_res_per_trail.append(1)
                else:
                    sad_res_per_trail.append(0)
                sad_res = 0
        sad_predicted_chances = sum(sad_res_per_trail) / len(sad_res_per_trail)
        accuracy = (happy_target_chances * len(happy_res_per_trail) + (1 - sad_predicted_chances)
                    * len(sad_res_per_trail)) / (len(happy_res_per_trail) + len(sad_res_per_trail))
        if happy_target_chances > sad_predicted_chances:
            result = 'Yes'
            # return 'Yes'
        else:
            result = 'no'
            # return 'no'

        mywin = visual.Window([420, 348], monitor="testMonitor", units="deg")
        start_block_win = visual.ImageStim(win=mywin, image=f'Pictures/gini.png')
        start_block_win.draw()
        mywin.flip()
        time.sleep(3)

        start_block_win = visual.TextStim(mywin, result,
                                          color=(1, 1, 1),
                                          colorSpace='rgb')
        start_block_win.draw()
        mywin.flip()
        time.sleep(10)
        mywin.close()
