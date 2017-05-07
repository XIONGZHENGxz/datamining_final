import numpy as np
from preprocess.preprocessing import preprocess

X, y, X_test = preprocess()
print X_test.shape
