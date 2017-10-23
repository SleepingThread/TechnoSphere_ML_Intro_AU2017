from sklearn.model_selection import KFold
import sklearn.model_selection
import sklearn.datasets
from sklearn import tree
import numpy as np

def stack_pred(estimator,X,y,Xt,k=3,method='predict'):
    """
    X and X train:
        X,Xt
    stack features for X and X train:
        sX, sXt
    k - amount of folds
    """
    sX = np.array([0.0 for i in range(0,len(X))])
    sXt = np.array([0.0 for i in range(0,len(Xt))])
    cv = KFold(n_splits=k,shuffle=True,random_state = 0)
    for train_ind, test_ind in cv.split(X):
        X_train, X_pred = X[train_ind], X[test_ind]
        y_train = y[train_ind]

        estimator.fit(X_train,y_train)
        #train feature fold
        trainff = estimator.predict(X_pred)
        sX[test_ind] = trainff
        #test feature
        testf = estimator.predict(Xt)
        sXt += testf
       
    sXt /= float(k)

    return (sX,sXt)

iris = sklearn.datasets.load_iris()
X_full,y_full = iris.data, iris.target
X, Xt,y,yt = sklearn.model_selection.train_test_split(X_full,y_full,test_size = 0.3, random_state = 0)

cltree = tree.DecisionTreeClassifier()

sX,sXt = stack_pred(cltree,X,y,Xt,k=3)

print 'Train feature: ',len(sX)
print sX
print 'Test feature: ',len(sXt)
print sXt
