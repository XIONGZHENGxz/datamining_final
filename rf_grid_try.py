import numpy as np
from model import model, index
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess.preprocessing import preprocess
from sklearn.model_selection import GridSearchCV

X, y, X_test = preprocess()
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

parameters = {
        'n_estimators' : [10, 20, 30, 40, 50],
        'min_samples_leaf' : [0.1, 0.2, 0.3, 0.4, 0.5],
        'max_features' : ('auto', 'sqrt', 'log2')
        }
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters)
clf.fit(X_train, y_train)
print "parameters: ", clf.cv_results_.keys()
print "score: ", clf.score(X_test, y_test)



