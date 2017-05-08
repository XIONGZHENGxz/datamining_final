import preprocessing
import pickle

preprocessing.plot_pca('pca.obj','pca_fig1.png','pca_fig2.png')
'''
in_file = open('pca.obj','rb')
pca = pickle.load(in_file)

tol = 0.85
count =0 
total = 0
for var in pca.explained_variance_ratio_:
	total+=var
	if total > tol:
		break
	count+=1


print count
'''
