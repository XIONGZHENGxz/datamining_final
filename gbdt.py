import numpy as np
from model import model, index
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
# from preprocess.preprocessing import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from fileIO import writeToFile
from model import index

class GBDT(model):
    def __init__(self):
        self.gbdt_classifier = GradientBoostingClassifier()
    
# -------- parameters ----
# learning_rate : float, optional (default=0.1)
# n_estimators : int (default=100)
# max_depth : integer, optional (default=3)
# min_samples_leaf : int, float, optional (default=1)
# max_features : int, float, string or None, optional (default=None)
# random_state : int, RandomState instance or None, optional (default=None)

    # def train(self, data_train):
    def gridSearch(self, X_train, y_train):
        # parameters = {'learning_rate' : np.arange(.1, 1, .1),
        #             'n_estimators' : np.arange(10, 40, 10),
        #             'max_depth' : 3,
        #             'min_samples_leaf' : np.arange(1, 100, 30),
        #             'max_features' : None,
        #             'random_stae' : 233
        #             }
        # clf = GridSearchCV(svr, parameters)
        # f1_scorer = make_scorer(f1_score)

        # X = data_train[:, 1:]
        # y = data_train[:, 0]
        # y = y.astype(int)
        y_train = y_train.astype(int)
        
        best_f1score, best_clf = 0, None
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
        # for n_estimators in np.arange(5, 50, 10):
        for n_estimators in [25]:
            # for learning_rate in np.arange(.1,1,.1):
            for learning_rate in [.3]:
                # for min_samples_leaf in np.arange(1, 21, 10):
                # for min_samples_leaf in np.arange(1, 40, 10):
                for min_samples_leaf in [1]:
                    # for max_features in ('auto', 'sqrt', 'log2'):
                    for max_features in [None]:
                        gbdt = GradientBoostingClassifier(
                               n_estimators = n_estimators,
                               learning_rate = learning_rate,
                               min_samples_leaf = min_samples_leaf,
                               max_features = max_features
                               )
                        # rf = RandomForestClassifier()
                        gbdt.fit(X_train, y_train)
                        y_pred = gbdt.predict(X_test)
                        curr_score = f1_score(y_test, y_pred)
                        if(curr_score >= best_f1score):
                            best_f1score = curr_score
                            best_clf = gbdt
                        print "n_estimators: ", n_estimators, " learning rate: ", learning_rate, "min_samples_leaf:", min_samples_leaf,"f1 score: ", curr_score
        

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
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        self.X_train, self.y_train = X_train, y_train.astype(int)
        self.X_train_lr, self.y_train_lr = X_train_lr, y_train_lr.astype(int)   # for plotting ROC curve
        self.gbdt_classifier = self.gridSearch(X_train, y_train)

    def predict(self, X_pred):
        y_pred = self.gbdt_classifier.predict(X_pred)
        return y_pred

    #def evaluate(self, data_test):
    def evaluate(self, data_test):
        X_test = data_test[:,1:]
        y_test = data_test[:,0]
        y_test = y_test.astype(int)
        y_pred = self.gbdt_classifier.predict(X_test)
        f1score = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        #---------calculate roc-----------
        gbdt = self.gbdt_classifier
        gbdt_enc = OneHotEncoder()
        gbdt_lm = LogisticRegression()
        # print "self.X_train.shape: ", self.X_train.shape
        # print np.squeeze(gbdt.apply(self.X_train)).shape
        # print gbdt.apply(self.X_train_lr).shape
        # print gbdt.apply(X_test).shape

        gbdt_enc.fit(np.squeeze( gbdt.apply(self.X_train) ))
        gbdt_lm.fit(gbdt_enc.transform(np.squeeze( gbdt.apply(self.X_train_lr))), self.y_train_lr)
        y_pred_gbdt_lm = gbdt_lm.predict_proba(gbdt_enc.transform(np.squeeze( gbdt.apply(X_test) )))[:, 1]
        fpr_gbdt_lm, tpr_gbdt_lm, _ = roc_curve(y_test, y_pred_gbdt_lm)
        self.fpr_gbdt_lm, self.tpr_gbdt_lm = fpr_gbdt_lm, tpr_gbdt_lm
        auc_score = auc(fpr_gbdt_lm, tpr_gbdt_lm)
        # print "auc score; ", auc_score
        eval_index = index(auc_score, f1score)
        self.roc_curve("roc_gbdt")
        return eval_index

    def roc_curve(self, file_path):
        print 'ploting roc curve >>>>'
        plt.figure()
        plt.plot(self.fpr_gbdt_lm,self.tpr_gbdt_lm,color='blue',label='ROC curve')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.title('Receiver operating characteristic')
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.savefig(file_path)


# data = np.load('preprocess/data.npy')
# X, y, X_submit = preprocess()
X = np.load("preprocess/X_train_labeled.npy")
X_submit = np.load("preprocess/X_test_labeled.npy")
y = np.load("preprocess/y.npy")


y = y.astype(int)
y = y.reshape(y.shape[0], 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
gbdt_classifier = GBDT()


data_train = np.hstack((y_train, X_train))
gbdt_classifier.train(data_train)

data_test = np.hstack((y_test, X_test))


# f1score, cm, auc_score = gbdt_classifier.evaluate(data_test)
# f1score, cm = gbdt_classifier.evaluate(data_test)
eval_index = gbdt_classifier.evaluate(data_test)

y_submit = gbdt_classifier.predict(X_submit)
# print "size of X_test: ", X_submit.shape
writeToFile('gbdt_result.csv', y_submit)

print "f1_score: ",  eval_index.F_score
# print "confusion matrix: "
# print cm

print "auc score: ", eval_index.AUC










