from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import pickle

clf = BernoulliNB()
X = pickle.load(open('body.data','r'))
Y = pickle.load(open('tags.data','r')).toarray()

clf.fit(X,Y)
