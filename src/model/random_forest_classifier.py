from sklearn.ensemble import RandomForestClassifier
from src.ultilities import *
from src.feature_engineering import *
import numpy as np
from collections import Counter
import os
import pickle


class RandomForestClassifier():

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=10)
        # self.params = {'n_estimators': [n for n in [10, 200, 500, 1000]]}
        # self.grid_search = GridSearchCV(self.clf, self.params, cv=5, n_jobs=-1)

    def train(self, x_train, y_train):

        pkl_file = 'rf_classifier.pkl'

        if os.path.isfile(pkl_file):

            print('Using pre-trained model ....')

        else:

            self.clf.fit(x_train, y_train)

            # save model:
            pickle.dump(self.clf, open(pkl_file, 'wb'))

            print("Finish training random forest")

    def get_label(self, x_test):

        # Load pre-trained model:
        pkl_file = 'rf_classifier.pkl'

        if os.path.isfile(pkl_file):

            loaded_model = pickle.load(open(pkl_file, 'rb'))
            predict_result = loaded_model.predict(x_test).tolist()

        else:

            predict_result = self.clf.predict(x_test).tolist()

        most_common_label, num_most_common_label = Counter(predict_result).most_common(1)[0]

        return most_common_label

