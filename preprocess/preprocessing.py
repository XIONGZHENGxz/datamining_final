import numpy as np
import math
import pandas as pd
import math
from sklearn.linear_model import Ridge
from sets import Set
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
import csv
import datetime

mean = [2005,4,6180.28,7385.99,8467.23,9964,6155.47,7382.22,8767.61,10128.55,6731.03,6731,1277]
m = {31:6731,33:1277}

keep = [5,6,7,8,9,10,11,12,14,15,16]  
factor = [3,6,7,8,9,10,11,12,15,16,17,26,27,28,29,30,32]
fact = [x-1 for x in factor]
numeric = [4,5,14,18,19,20,21,22,23,24,25,31,33]
nume = [x-1 for x in numeric]


def preprocess():
	data = np.load('data.npy')
	data_test = np.load('data_test.npy')
	data = np.delete(data,[2,13],axis=1)
	data_test = np.delete(data_test,[1,12],axis=1)
	y = data[:,1]
	X_test = data_test[:,1:]
	X = data[:,2:]
	return X,y,X_test

def normalize(data,nums):
	scaler = StandardScaler()
	data_new = data.copy()
	for i in nums:
		data_new[:,i] = scaler.fit_transform(data_new[:,i])
	return data_new

def save(input_path,test_path):
	data,data_test = fill_miss(input_path,test_path)
	print 'complete filling missing data...'
	data_new, label_encoders = label(data,factor)
	data_test = label_miss(data_new,data,fact,data_test,label_encoders,1,nume)
	print 'complete labeling test data...'
	print data_new.shape
	print data_test.shape
	np.save('data.npy',data_new)	
	np.save('data_test.npy',data_test)	

def getNB(sample,train,factor,numeric,flag):
	min_dist = 100000
	nb = -1

	for i in range(train.shape[0]):
		dist = 0
		for j in factor:
			if train[i,j+flag]!=sample[j]:
				dist+=1

		for j in numeric:
			dist+=abs(train[i,j+flag]-sample[j])

		if dist<min_dist:
			min_dist = dist
			nb = i

	return nb,min_dist
	

def conditional_mean_models(input_path,test_path):
	#----------------`--------train models----------------------#

	##columns we keep
	labels = [1,2,3,4,5,6,7,9,10]
	nums = [0,8]
	prices = range(18,26)
	good,miss = prep(input_path,0)
	good_test,miss_test = prep(test_path,1)
	total = np.vstack([good,miss])

	total_keep = total[:,keep]
	miss_test_keep = miss_test[:,[x-1 for x in keep]]

	## label training 
	total_keep_labeled,label_encoders = label(total_keep,labels)

	## label miss
	miss_test_keep = label_miss(total_keep_labeled,total_keep,labels,miss_test_keep,label_encoders,0,nums)
	print 'miss_test_keep,',miss_test_keep[0,:]

	##one hot encode 
	total_keep,encoder= encode(total_keep_labeled,labels)
	miss_test_keep = encoder.transform(miss_test_keep).toarray()

	print total_keep.shape
	train = total_keep[:good.shape[0],:]
	miss = total_keep[good.shape[0]:,:]
		
	models = []
	ridge = Ridge()
	for i in prices:
		y = total[:good.shape[0],i]
		ridge.fit(train,y)
		models.append(ridge)
	
	print 'complete building models...'
	return models,miss,miss_test_keep

def label_miss(total_keep_labeled,total_keep,labels,miss,label_encoders,index,nums):

	## get all labels
	labels_new = labels
	if index==1:
		labels_new = [x+1 for x in labels]

	allLabels = getAllLabels(total_keep,labels_new)
	total_keep_normalized = total_keep
	miss_normalized = miss
	if index==1:
		total_keep_normalized = normalize(total_keep_normalized,numeric)
	else:
		total_keep_normalized = normalize(total_keep_normalized,nums)
	
	miss_normalized = normalize(miss_normalized,nums)
	print 'miss...',miss[0,:]
	#---check new labels-----#
	count = 0
	for i in range(miss.shape[0]):
		nb = -1
		for j in range(len(labels)):
			if miss[i,labels[j]] not in allLabels[j]:
				nb,dist = getNB(miss_normalized[i,:],total_keep_normalized,labels,nums,index)
				break

		#-----if no new label-----#
		if nb == -1:
			for k in range(len(labels)):
				x = np.asarray(miss[i,labels[k]]).reshape(1, -1)[0,:]
				y = label_encoders[k].transform(x)
				miss[i,labels[k]] = np.asscalar(y)
		else:
			count+=1
			if index==0:
				miss[i,:] = total_keep_labeled[nb,:]
			else:
				miss[i,2:] = total_keep_labeled[nb,3:]

	print 'number of new labels...',count
	print 'miss..labeled...,',miss[0,:]
	return miss

def fill_miss(input_path,test_path):
	models,encoded_miss,encoded_miss_test= conditional_mean_models(input_path,test_path)
	print encoded_miss_test.shape
	good,miss = prep(input_path,0)
	good_test,miss_test = prep(test_path,1)
	
	for i in range(miss.shape[0]):
		X = encoded_miss[i,:]
		print X.shape
		for j in range(miss.shape[1]):
			if j>=18 and j<=25 and (miss[i,j]==0 or miss[i,j]==1):
				miss[i,j] = models[j-18].predict(X)

	for i in range(miss_test.shape[0]):
		X = encoded_miss_test[i,:]
		print X.shape
		for j in range(miss_test.shape[1]):
			if j>=17 and j<=24 and (miss_test[i,j]==0 or miss_test[i,j]==1 or math.isnan(miss_test[i,j])):
				miss_test[i,j] = models[j-17].predict(X)

	return np.vstack([good,miss]),np.vstack([good_test,miss_test])

def getAllLabels(data,factor):
	all_labels = []
	
	for i in factor:
		lb = Set()
		for j in range(data.shape[0]):
			if type(data[j,i]) is np.ndarray:
				data[j,i] = np.asscalar(data[j,i])
			lb.add(data[j,i])
		all_labels.append(lb)
	return all_labels

def label(data,factor):
	label_encoders = []
	data_new = data.copy()
	for i in factor:
		le = preprocessing.LabelEncoder()
		data_new[:,i] = le.fit_transform(data[:,i])
		label_encoders.append(le)
	return data_new,label_encoders

def encode(data,factor):
	onehot_enc = preprocessing.OneHotEncoder(categorical_features=factor)
	data = onehot_enc.fit_transform(data).toarray()
	return data,onehot_enc

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
			if j not in range(18-flag,26-flag) and type(df[i,j])==float and math.isnan(df[i,j]):
				df[i,j] = 'UNKNOWN'

		for j in range(h):
			if j>=18-flag and j<=25-flag and (df[i,j]==0 or df[i,j]==1 or math.isnan(df[i,j])):
				tmp = np.reshape(df[i,:],[1,df[i,:].shape[0]])
				miss = np.append(miss,tmp,axis=0)
				ind.append(i)
				break

	good = np.delete(df,ind,axis=0)
	return good,miss
def save_csv(test_path,train_path):
	X,y,X_test = preprocess()
	for i in range(X_test.shape[0]):
		for j in range(X_test.shape[1]):
			if type(X_test[i,j]) is np.ndarray:
				X_test[i,j] =np.asscalar(X_test[i,j])
			if type(X[i,j]) is np.ndarray:
				X[i,j] =np.asscalar(X[i,j])
	write_csv(test_path,X_test)	
	write_csv(train_path,X)	

def write_csv(file_path,data):

	with open(file_path,'wb') as f:
		w = csv.writer(f, delimiter=',')
		for i in range(data.shape[0]):
			w.writerow(data[i,:])

train = 'training.csv'
test = 'test.csv'

#conditional_mean_models(train,test)
#save(train,test)
save_csv('X_test.csv','X_train.csv')
