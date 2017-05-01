import numpy as np
import math
import pandas as pd
import math
from sklearn.linear_model import Ridge
from sets import Set
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, scale
import csv
import datetime

mean = [6180.28,7385.99,8467.23,9964,6155.47,7382.22,8767.61,10128.55,6731.03]

keep = [5,6,7,8,9,10,11,12,14,15,16]  
factor = [3,6,7,8,9,10,11,12,15,16,17,26,27,28,29,30,32]
numeric = [4,5,14,18,19,20,21,22,23,24,25,31,33]

def preprocess():
	data = np.load('data.npy')
	print data.shape
	data = np.delete(data,[2,13],axis=1)
	print data.shape
#	scaler = preprocessing.StandardScaler()
	y = data[:,1]
	X = data[:,2:]
	#X = scaler.fit_transform(X)
	return X,y
'''
def normalize(data,test_path):

'''			
def save(input_path,test_path):
	data = fill_miss(input_path,test_path)
	data,_  = label(data,factor)
	print data.shape
	np.save('data.npy',data)	

def getNB(sample,train):
	min_dist = 100000
	nb = -1
	for i in range(train.shape[0]):
		dist = 0

		for j in factor:
			if train[i,j]!=sample[j]:
				dist+=1
		for j in numeric:
			dist+=abs(train[i,j]-sample[j])

		if dist<min_dist:
			min_dist = dist
			nb = i
	return nb
	

def conditional_mean_models(input_path,test_path):
	#----------------`--------train models----------------------#

	##columns we keep
	labels = [1,2,3,4,5,6,7,9,10]
	prices = range(18,26)
	good,miss = prep(input_path,0)
	good_test,miss_test = prep(test_path,1)
	total = np.vstack([good,miss])

	total_keep = total[:,keep]
	miss_test_keep = miss_test[:,[x-1 for x in keep]]

	##do label
	total_keep,label_encoders = label(total_keep,labels)
	'''
	for i in range(len(labels)):
		miss_test_keep[labels[i]-1,:] = label_encoders[i].transform(miss_test_keep[labels[i]-1,:]) 
	'''
	##one hot encode 
	total_keep,encoder= encode(total_keep,labels)
	'''
	miss_test_keep = encoder.transform(miss_test_keep).toarray()
	'''
	print total_keep.shape
	train = total_keep[:good.shape[0],:]
	miss = total_keep[good.shape[0]:,:]
		
	models = []
	ridge = Ridge()
	for i in prices:
		y = total[:good.shape[0],i]
		ridge.fit(train,y)
		models.append(ridge)

	return models,miss

def fill_miss(input_path,test_path):
	models,encoded_miss = conditional_mean_models(input_path,test_path)
	good,miss = prep(input_path,0)
	for i in range(miss.shape[0]):
		X = encoded_miss[i,:]
		for j in range(miss.shape[1]):
			if j>=18 and j<=25 and (miss[i,j]==0 or miss[i,j]==1):
				miss[i,j] = models[j-18].predict(X)

	return np.vstack([good,miss])

def fill_miss_test(input_path,train):
	good,miss = prep(input_path)
	##label miss
	labeled_miss = miss
	for i in range(miss.shape[0]):
		nb = -1
		#---check new labels-----#
		for j in len(factor):
			if miss[i,factor[j]] not in all_labels[j]:
				nb = getNB(miss[i,:],train)
				break
		#-----if no new label-----#
		if nb == -1:
			for k in len(factor):
				labeled_miss[i,fator[k]-1] = label_encoders[k].transform(miss[i,factor[k]-1])
			for k in range(17,25):
				if miss[i,j]==0 or miss[i,j]==1:
					labeled_miss[i,j] = models[k-17].predict()
		else:
			labeled_miss[i,:] = train[nb,:]
			
	label(miss)
	for i in range(miss.shape[0]):
		X = encoded_miss[i,:]
		for j in range(miss.shape[1]):
			if j>=18 and j<=25 and (miss[i,j]==0 or miss[i,j]==1):
				miss[i,j] = models[j-18].predict(X)
	return np.vstack([good,miss])

def label(data,factor):
	label_encoders = []

	for i in factor:
		le = preprocessing.LabelEncoder()
		data[:,i] = le.fit_transform(data[:,i])
		label_encoders.append(le)
	return data,label_encoders

def encode(data,factor):
	onehot_enc = preprocessing.OneHotEncoder(categorical_features=factor)
	data = onehot_enc.fit_transform(data).toarray()
	return data,onehot_enc

def preprocess_train():
	#------------------------------load data -------------------------
	preprocess('training.csv','training_new.csv',2)
	df_train = pd.read_csv("training_new.csv",delimiter=',')
	data_train = df_train.as_matrix()
	data_train = data_train[:,1:]
	data_train = np.delete(data_train,[4,12],axis=1)#delete VehicleAge column and WheelType column
	y_train=data_train[:,0]
	X_train=data_train[:,1:]
	print "X_train set shape: " , X_train.shape
	print "y_train shape: ", y_train.shape

	preprocess('test.csv','test_new.csv',1)
	df_test = pd.read_csv("test_new.csv",delimiter=',')
	data_test = df_test.as_matrix()
	data_test = data_test[:,1:]
	data_test = np.delete(data_test,[3,11],axis=1)#delete VehicleAge column and WheelType column
	X_test = data_test
	# print "test set shape: ", X_test.shape

	X_total = np.vstack((X_train, X_test))
	# print "total shape: ", X_total.shape

	#-------------------------deal with missing numeric data


	for i in range(X_total.shape[0]):
		for j in range(X_total.shape[1]):
			if j>=14 and j<=21 and (X_total[i,j] == 0 or X_total[i,j]==1 or (type(X_total[i,j])==float and math.isnan(X_total[i,j]))):
				X_total[i,j] = mean[j-14]
			if j==27 and X_total[i,j]==1:
				X_total[i,j] = mean[-1]

	# ------------------------encode discrete data ---------------------------
	factor = [1,3,4,5,6,7,8,9,11,12,13,22,23,24,25,26,28]
	onehot_enc = preprocessing.OneHotEncoder(categorical_features=factor)
	for i in range(X_total.shape[0]):
		for j in factor:
			if type(X_total[i,j])==float and math.isnan(X_total[i,j]):
				X_total[i,j]=100000000

	for i in factor:
		le = preprocessing.LabelEncoder()
		X_total[:,i] = le.fit_transform(X_total[:,i])

	np.save("labeled_x.npy",X_total[0:X_train.shape[0],:])
	X_total = onehot_enc.fit_transform(X_total).toarray()
	print X_total.shape		

	# ---------------split dataset to get origin training and test set --------
	X_train_encoded = X_total[0 : X_train.shape[0], :]
	X_test_encoded = X_total[X_train.shape[0] : X_train.shape[0] + X_test.shape[0], :]


	#----------------- try normalize before pca ----------
	X_train_encoded = normalize(X_train_encoded)
	X_test_encoded = normalize(X_test_encoded)
	return X_train_encoded,X_test_encoded,y_train
'''
def processDate(row,index):
	date = row[index].split('/')
	if len(date) == 3:
		d1 = datetime.datetime(int(date[2]), int(date[0]), int(date[1]))
		d2 = datetime.datetime.now()
		row[index] = (d2-d1).days
	return row
'''

#----find out numeric missing-----#
# return integral rows and missing rows 

def prep(input_path,flag):
	df = pd.read_csv(input_path,delimiter=',')
	df = df.as_matrix()
	miss = np.empty([0,df.shape[1]])
	w = df.shape[0]
	h = df.shape[1]
	ind = []
	for i in range(w):
		for j in range(h):
			if type(df[i,j])==float and math.isnan(df[i,j]):
				df[i,j] = 'UNKNOWN'
		for j in range(h):
			if j>=18-flag and j<=25-flag and (df[i,j]==0 or df[i,j]==1):
				tmp = np.reshape(df[i,:],[1,df[i,:].shape[0]])
				miss = np.append(miss,tmp,axis=0)
				ind.append(i)
				break

	good = np.delete(df,ind,axis=0)
	return good,miss
save('training.csv','test.csv')
X,y = preprocess()
