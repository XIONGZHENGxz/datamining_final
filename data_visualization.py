import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('./preprocess')
train = np.load('X_train_labeled.npy')
y = np.load('y.npy')

#----------vehicle age--------#
def visual_VehAge():
	good = np.zeros(10).tolist()
	bad = np.zeros(10).tolist()

	for i in range(y.shape[0]):
		ind = train[i,2]
		if y[i] == 1:
			bad[ind]+=1
		else:
			good[ind]+=1
	sum_good = sum(good)
	sum_bad = sum(bad)

	for i in range(len(good)):
		good[i]	= good[i]/sum_good
		bad[i] = bad[i]/sum_bad
	
	x = range(10)

	width = 0.5
	pos_good = [e-0.25 for e in x]
	pos_bad = [e+0.25 for e in x]
	plt.bar(pos_good,good,width,color = 'blue')
	plt.bar(pos_bad,bad,width,color = 'red')
	plt.legend(['Goodbuy','Badbuy'])
	plt.title('Scaled Vehicle Age')
	plt.xlabel('Vehicle Age')
	plt.ylabel('Ratio of Badbuy and Goodbuy')
	plt.savefig('VehicleAge.png')

# ----------wheeltype--------#
def visual_WheelType():
	plt.figure()
	good = np.zeros(4).tolist()
	bad = np.zeros(4).tolist()
	for i in range(y.shape[0]):
		ind = train[i,9]-1
		if y[i] == 1:
			bad[ind]+=1
		else:
			good[ind]+=1

	sum_good = sum(good)
	sum_bad = sum(bad)

	for i in range(len(good)):
		good[i]	= good[i]/sum_good
		bad[i] = bad[i]/sum_bad
	
	x = range(1,5)

	width = 0.5
	pos_good = [e-0.25 for e in x]
	pos_bad = [e+0.25 for e in x]
	plt.bar(pos_good,good,width,color = 'blue')
	plt.bar(pos_bad,bad,width,color = 'red')
	plt.legend(['Goodbuy','Badbuy'])
	plt.title('Scaled WheelTypeId')
	plt.xlabel('WheelTypeId')
	plt.ylabel('Ratio of Badbuy and Goodbuy')
	plt.xticks([1,2,3,4])
	plt.savefig('WheelType.png')
	
visual_VehAge()
visual_WheelType()
