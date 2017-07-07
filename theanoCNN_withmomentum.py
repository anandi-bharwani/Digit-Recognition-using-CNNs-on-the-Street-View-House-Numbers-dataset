#Best error rate: 0.145

import numpy as np
import theano.tensor as T
from util import y2indicator, error_rate, get_train_data, get_test_data, relu
import theano
from theano.tensor.nnet import conv2d
import theano.tensor.signal.pool as pool

def reshape(X):
	# input is (32, 32, 3, N)
	# ouput is (N, 3, 32, 32)

	N = X.shape[-1]
	out = np.zeros((N, 3, 32, 32))
	for i in range(N):
		for j in range(3):
			out[i,j,:,:] = X[:,:,j,i]
	return (out/255).astype(np.float32)

def convpool(X, W, b, poolsz=(2,2)):
	conv = conv2d(X,W)						#Convolution
	max_pool = pool.pool_2d(conv, ws=poolsz, ignore_border=True)	#Max-pooling
	return relu(max_pool +b.dimshuffle('x', 0, 'x', 'x'))		#Non-linearity

def init_filter(shape, poolsz):
	w = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:])/np.prod(poolsz))
	return w

def main():
	#Get train and test data
	XTrain, YTrain = get_train_data()
	YTrain_ind = y2indicator(YTrain)
	XTrain = reshape(XTrain)
	XTest, YTest = get_test_data()
	YTest_ind = y2indicator(YTest)
	XTest = reshape(XTest)

	N,K = YTrain_ind.shape
	M=100
	lr = np.float32(0.000001)
	reg = np.float32(0.01)
	mu = np.float32(0.99)
	poolsize = (2,2)
	batch_sz = 500
	no_batches = int(N/batch_sz)

	#Initial random weight values
	W1_shape = (20, 3, 5, 5)
	W1_init = init_filter(W1_shape, poolsize)
	b1_init = np.zeros([W1_shape[0]])

	W2_shape = (50, 20, 5, 5)
	W2_init = init_filter(W2_shape, poolsize)
	b2_init = np.zeros([W2_shape[0]])

	W3_init = np.random.randn(W2_shape[0]*5*5, M)/np.sqrt(W2_shape[0]*5*5 + M)
	b3_init = np.zeros([M])

	W4_init = np.random.randn(M,K)/np.sqrt(M+K)
	b4_init = np.zeros([K])
	
	#Create theano variables
	X = T.tensor4('X', dtype='float32')			#inputs
	Y = T.matrix('Y')
	W1 = theano.shared(W1_init.astype(np.float32), 'W1')		#Weights
	b1 = theano.shared(b1_init.astype(np.float32), 'b1')
	W2 = theano.shared(W2_init.astype(np.float32), 'W2')
	b2 = theano.shared(b2_init.astype(np.float32), 'b2')
	W3 = theano.shared(W3_init.astype(np.float32), 'W3')
	b3 = theano.shared(b3_init.astype(np.float32), 'b3')
	W4 = theano.shared(W4_init.astype(np.float32), 'W4')
	b4 = theano.shared(b4_init.astype(np.float32), 'b4')

	dW1 = theano.shared(np.zeros(W1_init.shape, dtype=np.float32))	#Momentum variables
	db1 = theano.shared(np.zeros(b1_init.shape, dtype=np.float32))
	dW2 = theano.shared(np.zeros(W2_init.shape, dtype=np.float32))
	db2 = theano.shared(np.zeros(b2_init.shape, dtype=np.float32))
	dW3 = theano.shared(np.zeros(W3_init.shape, dtype=np.float32))
	db3 = theano.shared(np.zeros(b3_init.shape, dtype=np.float32))
	dW4 = theano.shared(np.zeros(W4_init.shape, dtype=np.float32))
	db4 = theano.shared(np.zeros(b4_init.shape, dtype=np.float32))

	#Forward prop equations
	Z1 = convpool(X, W1, b1)			#2 Conv-pool layer
	Z2 = convpool(Z1, W2, b2)
	Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)		#Fully connected NN
	P = T.nnet.softmax(Z3.dot(W4) + b4)

	#Cost and prediction equations
	params = (W1, b1, W2, b2, W3, b3, W4, b4)
	reg_cost = reg*np.sum([(param*param).sum() for param in params])
	cost = (Y * T.log(P)).sum() + reg_cost
	pred = T.argmax(P, axis=1)

	#Update Weights
	W1_update = W1 + mu*dW1 + lr*T.grad(cost, W1)
	b1_update = b1 + mu*db1 + lr*T.grad(cost,b1)
	W2_update = W2 + mu*dW2 + lr*T.grad(cost, W2)
	b2_update = b2 + mu*db2 + lr*T.grad(cost,b2)
	W3_update = W3 + mu*dW3 + lr*T.grad(cost, W3)
	b3_update = b3 + mu*db3 + lr*T.grad(cost,b3)
	W4_update = W4 + mu*dW4 + lr*T.grad(cost, W4)
	b4_update = b4 + mu*db4 + lr*T.grad(cost,b4)

	#Gradient updates for momentum
	dW1_update = mu*dW1 + lr*T.grad(cost, W1)
	db1_update = mu*db1 + lr*T.grad(cost, b1)
	dW2_update = mu*dW2 + lr*T.grad(cost, W2)
	db2_update = mu*db2 + lr*T.grad(cost, b2)
	dW3_update = mu*dW3 + lr*T.grad(cost, W3)
	db3_update = mu*db3 + lr*T.grad(cost, b3)
	dW4_update = mu*dW4 + lr*T.grad(cost, W4)
	db4_update = mu*db4 + lr*T.grad(cost, b4)

	#Train function
	train = theano.function(
		inputs=[X,Y],
		updates=[ (W1, W1_update),
			(b1, b1_update),
			(W2, W2_update),
			(b2, b2_update),
			(W3, W3_update),
			(b3, b3_update),
			(W4, W4_update),
			(b4, b4_update),
			(dW1, dW1_update),
			(db1, db1_update),
			(dW2, dW2_update),
			(db2, db2_update),
			(dW3, dW3_update),
			(db3, db3_update),
			(dW4, dW4_update),
			(db4, db4_update),
		 ])

	#Get cost and prediction function
	get_res = theano.function(
		inputs=[X,Y],
		outputs=[cost,pred])

	#Run batch gradient descent
	costs = []
	for i in range(400):
		for n in range(no_batches):
			#get current batches
			XBatch = XTrain[n*batch_sz:(n*batch_sz + batch_sz), :]
			YBatch_ind = YTrain_ind[n*batch_sz:(n*batch_sz + batch_sz), :]
			#Forward prop
			train(XBatch, YBatch_ind)

			if(n%200 == 0):
				#YBatch = YTrain[n*batch_sz:(n*batch_sz + batch_sz)]
				c, P = get_res(XTest, YTest_ind)
				er = error_rate(P, YTest)	
				print("Iteration: ", i, "Cost: ", c, "Error rate: ", er)
				
#		er = 0
#		cost =0
#		for n in range(no_batches):
			#get current batches
#			XBatch = XTest[n*batch_sz:(n*batch_sz + batch_sz), :]
#			YBatch_ind = YTest_ind[n*batch_sz:(n*batch_sz + batch_sz), :]
#			YBatch = YTest[n*batch_sz:(n*batch_sz + batch_sz)]
#			c, P = get_res(XBatch, YBatch_ind)
#			er += error_rate(P, YBatch)	
#			cost+=c
#		print("Iteration: ", i, "Cost: ", cost, "Error rate: ", er)

if __name__=='__main__':
	main()
