import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from preprocess.preprocessing import preprocess
from sklearn.model_selection import train_test_split

X, y, X_test = preprocess()
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print f1_score(y_test, y_pred)
