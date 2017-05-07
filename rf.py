import numpy as np
from model import model, index
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from preprocess.preprocessing import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from fileIO import writeToFile

class RF(model):
    def __init__(self):
        self.rf_classifier = RandomForestClassifier()
    
    # def train(self, data_train):
    def gridSearch(self, X_train, y_train):
        X = data_train[:, 1:]
        y = data_train[:, 0]
        y = y.astype(int)
        best_f1score, best_clf = 0, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        for n_estimators in [10, 20, 30, 40, 50]:
            for min_samples_leaf in np.arange(1, 21, 10):
                for max_features in ('auto', 'sqrt', 'log2'):
                    #rf = RandomForestClassifier(
                    #        n_estimators = n_estimators,
                    #        min_samples_leaf = min_samples_leaf,
                    #        max_features = max_features
                    #        )
                    rf = RandomForestClassifier()
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    curr_score = f1_score(y_test, y_pred)
                    if(curr_score >= best_f1score):
                        best_f1score = curr_score
                        best_clf = rf
                        print "get f1 score: ", curr_score

        # my_scorer = make_scorer(f1_score)
        # rf = RandomForestClassifier()
        # clf = GridSearchCV(rf, parameters, scoring=my_scorer)
        # clf.fit(X_train, y_train)
        # return clf 
        return best_clf

    
    def train(self, data_train):
        X_train = data_train[:, 1:]
        y_train = data_train[:, 0]
        # self.rf_classifier.fit(X_train, y_train)
        self.rf_classifier = self.gridSearch(X_train, y_train)

    def predict(self, X_pred):
        y_pred = self.rf_classifier.predict(X_pred)
        return y_pred

    #def evaluate(self, data_test):
    def evaluate(self, data_test):
        X_test = data_test[:,1:]
        y_test = data_test[:,0]
        y_test = y_test.astype(int)
        y_pred = self.rf_classifier.predict(X_test)
        f1score = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return f1score, cm


# data = np.load('preprocess/data.npy')
X, y, X_submit = preprocess()
y = y.astype(int)
y = y.reshape(y.shape[0], 1)
# data = data[:,1:]   #the first col is line number
# X_train, X_test, y_train, y_test = train_test_split(data[:,1:], data[:,[0]], test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf_classifier = RF()
# print y_train.shape, X_train.shape
# rf_classifier.train(np.hstack((y_train, X_train)))

data_train = np.hstack((y_train, X_train))
rf_classifier.train(data_train)

data_test = np.hstack((y_test, X_test))
# rf_classifier.train(X_train, y_train)
# score = rf_classifier.evaluate(np.hstack((y_test, X_test)))
f1score, cm = rf_classifier.evaluate(data_test)

y_submit = rf_classifier.predict(X_submit)
print "size of X_test: ", X_submit.shape
writeToFile('result.csv', y_submit)

print "f1_score: ",  f1score
print "confusion matrix: "
print cm







