i
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search

import scipy.io
mat = scipy.io.loadmat('ex6data3_1.mat')
X = np.array(mat['X'])
y = np.array(mat['y'])

plt.figure()
y_index_0 = np.nonzero(y == 0)[0]
y_index_1 = np.nonzero(y == 1)[0]
 
plt.plot(X[y_index_0, 0], X[y_index_0, 1], 'o')
plt.plot(X[y_index_1, 0], X[y_index_1, 1], '+')
plt.xlim(np.min(X[:, 0]), np.max(X[:, 0])* 1.1)
plt.ylim(np.min(X[:, 1]), np.max(X[:, 1])* 1.1)

clf = svm.SVC(C=1000, kernel='rbf')

clf.fit(X, np.ravel(y))

x1_grid = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.01)
x2_grid = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.01)

xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
z = np.zeros(xx1.shape)
for i1 in range(xx1.shape[0]):
    for i2 in range(xx1.shape[1]):
        z[i1, i2] = clf.predict([xx1[i1, i2], xx2[i1, i2]])
    
plt.contourf(x1_grid, x2_grid, z, levels=[0, 0], linewidths=2)

cv = cross_validation.KFold(X.shape[0], n_folds = 6)

C=np.logspace(-1, 1, 10)
param_grid=dict()
param_grid['C'] = C
gs = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv)
gs.fit(X, np.ravel(y))
print gs.best_params_
print gs.best_estimator_.C
print gs.best_score_
