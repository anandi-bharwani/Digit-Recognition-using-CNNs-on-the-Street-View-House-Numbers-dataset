import numpy as np
from scipy.io import loadmat

def relu(X):
	return X * (X>0)

def error_rate(Y,T):
	return np.mean(Y!=T)

def y2indicator(Y):
	N = len(Y)
	K = len(set(Y))
	Y_ind = np.zeros([N,K])
	for i in range(N):
		Y_ind[i, Y[i]] = 1
	return Y_ind

def get_train_data():
	train_data = loadmat('train_32x32.mat')
	X = train_data['X']
	Y = (train_data['y'] - 1).flatten()
	return X[:,:,:,:20000],Y[:20000] 	#getting memory error if processing the entire data set


	
def get_test_data():
	test_data = loadmat('test_32x32.mat')
	X = test_data['X']
	Y = (test_data['y'] - 1).flatten()
	return X[:,:,:,:2000],Y[:2000]

#Y = np.array([[1,2],[3,4]])
#print(y2indicator(Y.flatten()))
