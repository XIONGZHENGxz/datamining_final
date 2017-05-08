import numpy as np
import matplotlib.pyplot as plt

# y_score is the score of positive class,title is the name of list chart, file_path is the file name to save this figure
def lift_chart(y_test,y_score,title,file_path):
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
	plt.title(title)
	plt.xlabel('Sample Size')
	plt.ylabel('Percentage of Badbuy Found')
	plt.scatter(taille,taille,marker='.',linewidths=0.05,color='blue')
	plt.scatter(taille,rappel,marker='.',color='red')
	plt.savefig(file_path)
