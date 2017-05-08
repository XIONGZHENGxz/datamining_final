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
	
# ----------odometer--------#
def visual_Odo():
	plt.figure()
	x = np.linspace(10000,120000,12)
	gap = 10000
	good = np.zeros(12).tolist()
	bad = np.zeros(12).tolist()
	print x
	for i in range(y.shape[0]):
		ind = train[i,10]/gap
		ind = int(ind)
		print ind
		if y[i] == 1:
			bad[ind]+=1
		else:
			good[ind]+=1

	sum_good = sum(good)
	sum_bad = sum(bad)
	for i in range(len(good)):
		good[i]	= good[i]/sum_good
		bad[i] = bad[i]/sum_bad

	print good,bad
	width = 2500
	pos_good = [e-7500 for e in x]
	pos_bad = [e-2500 for e in x]
	print pos_good
	plt.bar(pos_good,good,width,color = 'blue')
	plt.bar(pos_bad,bad,width,color = 'red')
	plt.legend(['Goodbuy','Badbuy'])
	plt.title('Scaled Odometer')
	plt.xlabel('Odometer')
	plt.ylabel('Ratio of Badbuy and Goodbuy')
	plt.xticks([int(e) for e in x ])
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.savefig('Odometer.png')


def visual_AAAP():
	plt.figure()
	x = np.linspace(0,35000,8)
	gap = 5000
	good = np.zeros(8).tolist()
	bad = np.zeros(8).tolist()
	for i in range(y.shape[0]):
		ind = train[i,14]/gap
		ind = int(ind)
		if y[i] == 1:
			bad[ind]+=1
		else:
			good[ind]+=1

	sum_good = sum(good)
	sum_bad = sum(bad)
	r = []
	for i in range(len(good)):
		good[i] = good[i]/sum_good
		bad[i] = bad[i]/sum_bad

	width = 1000
	pos_good = [e+2000 for e in x]
	pos_bad = [e+3000 for e in x]
	#pos = [e+2500 for e in x]
	plt.bar(pos_good,good,width,color = 'blue')
	plt.bar(pos_bad,bad,width,color = 'red')
	#plt.bar(pos,r,width,color = 'red')
	plt.legend(['Goodbuy', 'Badbuy'])
	plt.title('Scaled Price')
	plt.ylabel('Ratio of Badbuy and Goodbuy')
	plt.xlabel('MMRAcquisitionAuctionAveragePrice')
	plt.savefig('AAAP.png')
	
def visual_CAAP():
	plt.figure()
	x = np.linspace(0,35000,8)
	gap = 5000
	good = np.zeros(8).tolist()
	bad = np.zeros(8).tolist()
	for i in range(y.shape[0]):
		ind = train[i,18]/gap
		ind = int(ind)
		if y[i] == 1:
			bad[ind]+=1
		else:
			good[ind]+=1

	sum_good = sum(good)
	sum_bad = sum(bad)
	r = []
	for i in range(len(good)):
		good[i] = good[i]/sum_good
		bad[i] = bad[i]/sum_bad

	width = 1000
	pos_good = [e+2000 for e in x]
	pos_bad = [e+3000 for e in x]
	#pos = [e+2500 for e in x]
	plt.bar(pos_good,good,width,color = 'blue')
	plt.bar(pos_bad,bad,width,color = 'red')
	#plt.bar(pos,r,width,color = 'red')
	plt.legend(['Goodbuy', 'Badbuy'])
	plt.title('Scaled Price')
	plt.ylabel('Ratio of Badbuy and Goodbuy')
	plt.xlabel('MMRCurrentAuctionAveragePrice')
	plt.savefig('CAAP.png')

def scatter_odo_AAAP():
	plt.figure()
	plt.scatter(train[:,10],train[:,14])
	plt.title('Scatter Plot')
	plt.xlabel('Odometer')
	plt.ylabel('MMRCurrentAuctionAveragePrice')
	plt.savefig('Odo_appp.png')

def scatter_age_AAAP():
	plt.figure()
	plt.scatter(train[:,2],train[:,14])
	plt.title('Scatter Plot')
	plt.xlabel('Vehicle Age')
	plt.ylabel('MMRCurrentAuctionAveragePrice')
	plt.savefig('VehAge_appp.png')

def scatter_acq_prices():
	plt.figure()
	plt.title('Scatter Plot of Acquisition Price')
	plt.scatter(train[:,14], train[:,15])
	plt.xlabel('MMRAcquisitionAuctionAveragePrice')
	plt.ylabel('MMRAcquisitionAuctionCleanPrice')
	plt.savefig('aaap_aacp.png')

	plt.figure()
	plt.title('Scatter Plot of Acquisition Price')
	plt.scatter(train[:,15], train[:,16])
	plt.xlabel('MMRAcquisitionAuctionCleanPrice')
	plt.ylabel('MMRAcquisitionRetailAveragePrice')
	plt.savefig('aacp_arap.png')

	plt.figure()
	plt.title('Scatter Plot of Acquisition Price')
	plt.scatter(train[:,16], train[:,17])
	plt.xlabel('MMRAcquisitionRetailAveragePrice')
	plt.ylabel('MMRAcquisitionRetailCleanPrice')
	plt.savefig('arap_arcp.png')

	plt.figure()
	plt.title('Scatter Plot of Acquisition and Current Price')
	plt.scatter(train[:,17], train[:,18])
	plt.xlabel('MMRAcquisitionRetailCleanPrice')
	plt.ylabel('MMRCurrentAuctionAveragePrice')
	plt.savefig('arcp_caap.png')

	plt.figure()
	plt.scatter(train[:,18], train[:,19])
	plt.title('Scatter Plot of Current Price')
	plt.xlabel('MMRCurrentAuctionAveragePrice')
	plt.ylabel('MMRCurrentAuctionCleanPrice')
	plt.savefig('caap_cacp.png')

	plt.figure()
	plt.scatter(train[:,19], train[:,20])
	plt.title('Scatter Plot of Current Price')
	plt.xlabel('MMRCurrentAuctionCleanPrice')
	plt.ylabel('MMRCurrentRetailAveragePrice')
	plt.savefig('racp_crap.png')


	plt.figure()
	plt.scatter(train[:,20], train[:,21])
	plt.title('Scatter Plot of Current Price')
	plt.xlabel('MMRCurrenRetailAveragePrice')
	plt.ylabel('MMRCurrentRetailCleanPrice')
	plt.savefig('crap_crcp.png')

'''
visual_VehAge()
visual_WheelType()

visual_Odo()
visual_CAAP()
'''
scatter_odo_AAAP()
scatter_age_AAAP()
scatter_acq_prices()
