from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
clf = OneVsRestClassifier(LogisticRegression())
X = pickle.load(open('body.train.data','r'))
Y = pickle.load(open('tags.train.data','r'))
#X_test = pickle.load(open('body.test.0.data','r'))
last_row = Y.shape[0]
Y = [
		tuple(np.flatnonzero(Y.getrow(r).toarray()))
		for r in xrange(last_row)
	]
model = clf.fit(X,Y)
#print model.predict(X_test)
pickle.dump(model,open('model.data','w'),protocol=1)
