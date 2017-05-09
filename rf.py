import numpy as np
from model import model, index
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, auc, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
# from preprocess.preprocessing import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from fileIO import writeToFile
from model import index

class RF(model):
    def __init__(self):
        self.rf_classifier = RandomForestClassifier()
    
    # def train(self, data_train):
    def gridSearch(self, X_train, y_train):
        # X = data_train[:, 1:]
        # y = data_train[:, 0]
        # y = y.astype(int)
        y_train = y_train.astype(int)
        best_f1score, best_clf = 0, None
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
        # for n_estimators in [10, 20, 30, 40, 50]:
        for n_estimators in [50]:
            # for min_samples_leaf in np.arange(1, 21, 10):
            for min_samples_leaf in [20]:
                # for max_features in ('log2', 'sqrt', None):
                for max_features in [.5]:
                # for max_features in np.arange(.01, .9, .1):
                    for max_depth in [None]:
                        rf = RandomForestClassifier(
                               n_estimators = n_estimators,
                               min_samples_leaf = min_samples_leaf,
                               max_features = max_features,
                               class_weight = "balanced",
                               max_depth = max_depth
                               )
                        # rf = RandomForestClassifier()
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        curr_score = f1_score(y_test, y_pred)
                        if(curr_score >= best_f1score):
                            best_f1score = curr_score
                            best_clf = rf
                        print "max_features" , max_features, "max_depth: ", max_depth, " f1 score: ", curr_score
        

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
        # splitting data for plotting ROC curve
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        self.X_train, self.y_train = X_train, y_train.astype(int)
        self.X_train_lr, self.y_train_lr = X_train_lr, y_train_lr.astype(int)   # for plotting ROC curve
        self.rf_classifier = self.gridSearch(X_train, y_train)
        

    def predict(self, X_pred):
        y_pred = self.rf_classifier.predict(X_pred)
        return y_pred

    #def evaluate(self, data_test):
    def evaluate(self, data_test):  # including f1score, auc score, roc curve, lift chart
        X_test = data_test[:,1:]
        y_test = data_test[:,0]
        y_test = y_test.astype(int)
        y_pred = self.rf_classifier.predict(X_test)
        f1score = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # ---------- calculate auc
        rf = self.rf_classifier
        rf_enc = OneHotEncoder()
        rf_lm = LogisticRegression()
        rf_enc.fit(rf.apply(self.X_train))
        rf_lm.fit(rf_enc.transform(rf.apply(self.X_train_lr)), self.y_train_lr)
        y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
        fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
        self.fpr_rf_lm, self.tpr_rf_lm = fpr_rf_lm, tpr_rf_lm
        auc_score = auc(fpr_rf_lm, tpr_rf_lm)
        eval_index = index(auc_score, f1score)
        # print "y_pred_rf_lm: ", y_pred_rf_lm
        self.roc_curve("roc_rf")
        self.lift_chart(y_pred, y_pred_rf_lm, "liftchart of random forest classifier", "liftchart_rf")
        print "accuracy: ", accuracy_score(y_test, y_pred)
        return eval_index
        # return f1score, cm, auc_score

    # plot roc curve
    def roc_curve(self, file_path):
        print 'ploting roc curve >>>>'
        plt.figure()
        plt.plot(self.fpr_rf_lm,self.tpr_rf_lm,color='blue',label='ROC curve')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.title('Receiver operating characteristic')
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.savefig(file_path)

    def lift_chart(self, y_test,y_score,title,file_path):
        pos = y_test
        npos = np.sum(pos)
        index = np.argsort(y_score)
        index = index[::-1]
        sort_pos = pos[index]
        cpos = np.cumsum(sort_pos) 

        rappel = cpos/float(npos)

        n = y_test.shape[0]
        taille = np.arange(start=1,stop=n+1,step=1)

        taille = taille / float(n)
        import matplotlib.pyplot as plt
        #title and axis labels
        plt.figure()
        plt.title(title)
        plt.xlabel('Sample Size')
        plt.ylabel('Percentage of Badbuy Found')
        plt.scatter(taille,taille,marker='.',linewidths=0.05,color='blue')
        plt.scatter(taille,rappel,marker='.',color='red')
        plt.savefig(file_path)

        

# data = np.load('preprocess/data.npy')
# X, y, X_submit = preprocess()
X = np.load("preprocess/X_train_labeled.npy")
X_submit = np.load("preprocess/X_test_labeled.npy")
y = np.load("preprocess/y.npy")


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
# f1score, cm, auc_score = rf_classifier.evaluate(data_test)
eval_index = rf_classifier.evaluate(data_test)

y_submit = rf_classifier.predict(X_submit)
# print "size of X_test: ", X_submit.shape
writeToFile('rf_result.csv', y_submit)





print "f1_score: ",  eval_index.F_score
# print "confusion matrix: "
# print cm

print "auc score: ", eval_index.AUC








