import numpy as np
import os
import pickle 
import matplotlib.pyplot as plt
from model import model, index
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocess import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from fileIO import writeToFile

class log(model):
    def __init__(self):
        self.clf = LogisticRegression(class_weight = 'balanced')
    
    # def train(self, data_train):
    def gridSearch(self, X, y):
		print 'gridsearching >>>>'
		best_f1score, best_clf = 0, None
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
		print X_train.shape,y_train.shape
		params = {'C':range(8,9)}
		clf_best = GridSearchCV(self.clf,params,cv=5,scoring = 'roc_auc')
		clf_best.fit(X_train,y_train)
		best_C = clf_best.best_params_['C']
		return best_C
    
    def train(self, X, y):
		y = np.array(y,dtype='|S4')
		y = y.astype(np.int)
		print 'training >>>>'
		# best_C = self.gridSearch(X,y)
		self.clf = LogisticRegression(C=8,class_weight = 'balanced')
		self.clf.fit(X,y)

    def predict(self, X_pred):
		print 'predicting >>>>'
		y_pred = self.clf.predict(X_pred)
		return y_pred

    # def evaluate(self, data_test):
    def evaluate(self, X_test, y_test):
		print 'evaluating >>>>'
		y_test = y_test.astype(int)
		y_score = self.clf.decision_function(X_test)
		y_pred = self.clf.predict(X_test)
		f1score = f1_score(y_test, y_pred)
		auc = roc_auc_score(y_test, y_score)
		cm = confusion_matrix(y_test, y_pred)
		score = accuracy_score(y_test, y_pred)
		return f1score, cm, auc, score

	# plot roc curve
    def roc_curve(self, X_test, y_test,file_path):
		print 'ploting roc curve >>>>'
		y_score = self.clf.decision_function(X_test)
		fpr, tpr, _ = roc_curve(y_test,y_score)
		plt.figure()
		plt.plot(fpr,tpr,color='blue',label='ROC curve')
		plt.xlim([0.0,1.0])
		plt.ylim([0.0,1.0])
		plt.title('Receiver operating characteristic')
		plt.xlabel('False Positive')
		plt.ylabel('True Positive')
		plt.savefig(file_path)
'''
os.chdir('./preprocess')
X = np.load('X_train_pca.npy')
X_submit = np.load('X_test_pca.npy')
y = np.load('y.npy')
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
log_clf = log()
log_clf.train(X_train,y_train)
f1score, cm, auc, score = log_clf.evaluate(X_test,y_test)
y_submit = log_clf.predict(X_submit)
writeToFile('log_result.csv', y_submit)

print "f1_score: ",  f1score
print "confusion matrix: "
print cm

print 'auc: ',auc
print 'score: ',score
log_clf.roc_curve(X_test,y_test,'log_roc.png')
filename = 'log_model.asv'
out = open(filename,'wb')
pickle.dump(log_clf,out)



'''
