#0.28

import tensorflow as tf
import numpy as np
from util import error_rate, y2indicator, get_train_data, get_test_data


def reshape(X):
	#Input : (32, 32, 3, N)
	#Output: (N, 32, 32, 3)
	return X.transpose(3, 0, 1, 2)/255

def init_filter(shape, poolsz):
	W = np.random.randn(*shape)/np.sqrt(np.prod(shape[:2]) + np.prod(shape[:1])*shape[3]/np.prod(poolsz))
	return W

def convpool(X, W, b):
	conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
	pool = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	return tf.nn.relu(tf.nn.bias_add(pool, b))
	

def main():
	#Get train and test data
	XTrain, YTrain = get_train_data()
	YTrain_ind = y2indicator(YTrain)
	XTrain = reshape(XTrain)

	XTest, YTest = get_test_data()
	YTest_ind = y2indicator(YTest)
	XTest = reshape(XTest)

	N, K = YTrain_ind.shape
	lr = np.float32(0.001)
	mu = np.float32(0.99)
	reg = np.float32(0.01)
	poolsz = (2,2)
	M = 100
	batch_sz = 500
	no_batches = int(N/batch_sz)

	#Initial random weights
	W1_shape = (5, 5, 3, 20)
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros([W1_shape[3]])

	W2_shape = (5, 5, 25, 50)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros([W2_shape[3]])
	
	W3_init = np.random.randn(W2_shape[3]*8*8, M)/np.sqrt(W2_shape[3]*8*8 + M)
	b3_init = np.zeros([M])

	W4_init = np.random.randn(M, K)/np.sqrt(M+K)
	b4_init = np.zeros([K])

	#Tensorflow variables
	X = tf.placeholder(name='X', dtype='float32', shape=(batch_sz, 32, 32, 3))
	Y = tf.placeholder(name='Y', dtype='float32', shape=(batch_sz, K))
	W1 = tf.Variable(W1_init.astype(np.float32), name='W1')
	b1 = tf.Variable(b1_init.astype(np.float32), name='b1')
	W2 = tf.Variable(W2_init.astype(np.float32), name='W2')
	b2 = tf.Variable(b2_init.astype(np.float32), name='b2')
	W3 = tf.Variable(W3_init.astype(np.float32), name='W3')
	b3 = tf.Variable(b3_init.astype(np.float32), name='b3')
	W4 = tf.Variable(W4_init.astype(np.float32), name='W4')
	b4 = tf.Variable(b4_init.astype(np.float32), name='b4')
	
	#Forward prop
	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z2_shape = Z2.get_shape().as_list()
	Z2_flat = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
	Z3 = tf.nn.relu(tf.matmul(Z2_flat,W3) + b3)
	pY = tf.matmul(Z3,W4) + b4

	#Cost and prediction
	cost = 	tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=Y))

	#Train function
	train = tf.train.RMSPropOptimizer(lr, decay = 0.99, momentum = mu).minimize(cost)

	#Get prediction
	pred = tf.argmax(pY, axis=1)

	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		for i in range(100):
			for n in range(no_batches):
				#get current batches
				XBatch = XTrain[n*batch_sz:(n*batch_sz + batch_sz), :]
				YBatch_ind = YTrain_ind[n*batch_sz:(n*batch_sz + batch_sz), :]
				#Forward prop
				session.run(train, feed_dict={X:XBatch, Y:YBatch_ind})

				if(n%200 == 0):
					YBatch = YTrain[n*batch_sz:(n*batch_sz + batch_sz)]
					c = session.run(cost, feed_dict={X:XBatch,Y:YBatch_ind})
					P = session.run(pred, feed_dict={X:XBatch})
					er = error_rate(P, YBatch)
					print("Iteration: ", i, "Cost: ", c, "Error rate: ", er)
if __name__=='__main__':
	main()
