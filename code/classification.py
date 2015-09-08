import numpy as np
import pickle
import pylab as pl
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier


def classify(traindata, target):
	
	metrics = ['accuracy', 'precision', 'recall', 'f1']
	print " ,accuracy, precision, recall, f1"
	'''
	print 'gini,',
	clf = DecisionTreeClassifier(criterion='gini', random_state=0)
	for item in metrics:
		scores = cross_validation.cross_val_score(clf, traindata, target, cv=5, n_jobs=-1, scoring=item)
		print "%0.4f," % scores.mean(),
	print '\nentropy,',
	clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
	for item in metrics:
		scores = cross_validation.cross_val_score(clf, traindata, target, cv=5, n_jobs=-1, scoring=item)
		print "%0.4f," % scores.mean(),
	print '\nmax_features: sqrt,',
	clf = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', random_state=0)
	for item in metrics:
		scores = cross_validation.cross_val_score(clf, traindata, target, cv=5, n_jobs=-1, scoring=item)
		print "%0.4f," % scores.mean(),
	print '\nmax_features: log2,',
	clf = DecisionTreeClassifier(criterion='entropy', max_features='log2', random_state=0)
	for item in metrics:
		scores = cross_validation.cross_val_score(clf, traindata, target, cv=5, n_jobs=-1, scoring=item)
		print "%0.4f," % scores.mean(),
'''
	for i in xrange(2, 103, 10):
		print '\nmin_samples_split: ' + str(i) + ',',
		clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=i, random_state=0)
		for item in metrics:
			scores = cross_validation.cross_val_score(clf, traindata, target, cv=5, n_jobs=-1, scoring=item)
			print "%0.4f," % scores.mean(),
			
def main():
	print "loading training dataset features"
	traindata = np.asarray(pickle.load(open("traindata-allfeatures.list", "r"))).astype(np.float)
	print "loading class labels for training dataset"
	target = np.asarray(pickle.load(open("target.list", "r"))).astype(np.float)

	print 'Size of training:' + str(len(traindata))
	classify(traindata, target)
	
if  __name__ == '__main__':
	main()
	

