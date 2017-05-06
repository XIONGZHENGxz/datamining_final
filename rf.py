import numpy as np
from model import model, index
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from preprocess.preprocessing import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

class RF(model):
    def __init__(self):
        self.rf_classifier = RandomForestClassifier()
    
    # def train(self, data_train):
    def gridSearch(self, X_train, y_train):
        parameters = { 
            'n_estimators' : [10, 20, 30, 40, 50],
            'min_samples_leaf' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'max_features' : ('auto', 'sqrt', 'log2')
            }
        my_scorer = make_scorer(f1_score)
        rf = RandomForestClassifier()
        clf = GridSearchCV(rf, parameters, scoring=my_scorer)
        clf.fit(X_train, y_train)
        return clf 

    
    def train(self, X_train, y_train):
        # X_train = data_train[:, 1:]
        # y_train = data_train[:, 0]
        # self.rf_classifier.fit(X_train, y_train)
        self.rf_classifier = self.gridSearch(X_train, y_train)

    def predict(self, X_pred):
        y_pred = self.rf_classifier.predict(X_pred)
        return y_pred
    #def evaluate(self, data_test):
    def evaluate(self, X_test, y_test):
        # X_test = data_test[:,1:]
        # y_test = data_test[:,0]
        y_pred = self.rf_classifier.predict(X_test)
        return f1_score(y_test, y_pred)


# data = np.load('preprocess/data.npy')
X, y, X_test = preprocess()
y = y.astype(int)
# data = data[:,1:]   #the first col is line number
# X_train, X_test, y_train, y_test = train_test_split(data[:,1:], data[:,[0]], test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf_classifier = RF()
# print y_train.shape, X_train.shape
# rf_classifier.train(np.hstack((y_train, X_train)))
rf_classifier.train(X_train, y_train)
# score = rf_classifier.evaluate(np.hstack((y_test, X_test)))
score = rf_classifier.evaluate(X_test, y_test)
print score








