import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics 
from sklearn.ensemble import AdaBoostClassifier

def classifyrandomforest(traindata, target, maxdepth=None):
	
	metrics = ['accuracy', 'precision', 'recall', 'f1']
	print " ,accuracy, precision, recall, f1"

	for notrees in xrange(10,101,10):
	        clf = RandomForestClassifier(n_estimators=notrees, max_depth=maxdepth,min_samples_split=1, random_state=0, verbose=0)
                print str(notrees)+",",
		for item in metrics:
			scores = cross_validation.cross_val_score(clf, traindata, target, cv=5, n_jobs=-1, scoring=item)
			print "%0.4f," % scores.mean(),
                print "\n"

def classifyadaboost(traindata, target, maxdepth=None):
	metrics = ['accuracy', 'precision', 'recall', 'f1']
	print " ,accuracy, precision, recall, f1"
	clf = DecisionTreeClassifier(criterion='entropy', max_depth=maxdepth,random_state=0)
        bdt = AdaBoostClassifier(clf)
        for item in metrics:
		scores = cross_validation.cross_val_score(bdt, traindata, target, cv=5, n_jobs=-1, scoring=item)
		print "%0.4f," % scores.mean(),
        		
def main():
	print "loading training dataset features"
	traindata = np.asarray(pickle.load(open("data/traindata-allfeatures.list", "r"))).astype(np.float)
	print "loading class labels for training dataset"
	target = np.asarray(pickle.load(open("data/target.list", "r"))).astype(np.float)

	print 'Size of training:' + str(len(traindata))
	#classifyrandomforest(traindata, target)
        #classifyrandomforest(traindata,target,10)
        #print 'adaboost without maxdepth'
	#classifyadaboost(traindata,target)
        print 'adaboost with maxdepth'
	classifyadaboost(traindata,target,10)
	
		
if  __name__ == '__main__':
	main()
	

