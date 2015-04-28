import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
import scipy.io

def display_data(X, y, clf):
    
    plt.figure()
    y_index_0 = np.nonzero(y == 0)[0]
    y_index_1 = np.nonzero(y == 1)[0]
    plt.plot(X[y_index_0, 0], X[y_index_0, 1], 'o')
    plt.plot(X[y_index_1, 0], X[y_index_1, 1], '+')
    plt.xlim(np.min(X[:, 0]), np.max(X[:, 0])* 1.1)
    plt.ylim(np.min(X[:, 1]), np.max(X[:, 1])* 1.1)
    
    
    x1_grid = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.01)
    x2_grid = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.01)
    
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
    z = np.zeros(xx1.shape)
    for i1 in range(xx1.shape[0]):
        for i2 in range(xx1.shape[1]):
            z[i1, i2] = clf.predict([xx1[i1, i2], xx2[i1, i2]])
        
    plt.contourf(x1_grid, x2_grid, z, levels=[0, 0])
    
    plt.figure()
    plt.plot(C_vec, scores)
        

class SVC_1(svm.SVC):
    def score(self, X, y):
        #return np.mean(self.predict(X) == y)
        parent = super(svm.SVC, self).score(X, y)
        child =  np.mean(self.predict(X) == y)
        return child

        
if (__name__ == "__main__"):

    mat = open('/home/amir/git/Spam_filter/main.py','r')
    mat = scipy.io.loadmat('/home/amir/git/Spam_filter/ex6data3_1.mat')
    X = np.array(mat['X'])
    y = np.array(mat['y'])
    
    m = X.shape[0]
    rand_index = np.random.permutation(m)
    X_train = X[rand_index[:0.9 * m], :]
    X_test = X[rand_index[0.9 * m:], :]
    y_train = y[rand_index[:0.9 * m]]
    y_test = y[rand_index[0.9 * m:]]
    
    clf = SVC_1(C=1000, kernel='rbf')
    
    cv = cross_validation.KFold(X_train.shape[0], n_folds = 6)
    
    C_vec=np.logspace(-1, 1, 10)
    param_grid=dict()
    param_grid['C'] = C_vec
    gs = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv)
    gs.fit(X_train, np.ravel(y_train))
    print gs.best_params_
    print gs.best_estimator_.C
    print gs.best_score_
    
    scores = np.zeros(C_vec.shape)
    for i in range(len(C_vec)):
        for train_indices, test_indices in cv:
            print train_indices
            print test_indices
            print
            clf.set_params(C=C_vec[i])
            scores[i] = clf.fit(X_train[train_indices, :], np.ravel(y_train[train_indices])).score(X_train[test_indices, :], np.ravel(y_train[test_indices]))
                        
    print scores
    
    display_data(X_test, y_test, clf)
    plt.show()




    
    
    
