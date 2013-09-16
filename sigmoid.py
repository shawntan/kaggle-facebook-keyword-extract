import theano
import math
import pickle
import theano.tensor as T
import numpy         as np
import utils         as U
from theano import sparse
from scipy.sparse import csr_matrix
def shared_sparse(arr):
	data    = arr.data
	indices = arr.indices
	indptr  = arr.indptr
	shape   = np.array(arr.shape)
	return sparse.CSR(data,indices,indptr,shape)



if __name__ == "__main__":
	training_data   = shared_sparse(csr_matrix(np.eye(100)))

	#training_labels = pickle.load(open('tags.train.data','r'))
	
	W = U.create_shared(U.initial_weights(71165,26920))
	out  = theano.dot(training_data,W)
	f = theano.function(
			inputs = [],
			outputs = out
		)
	print f()






