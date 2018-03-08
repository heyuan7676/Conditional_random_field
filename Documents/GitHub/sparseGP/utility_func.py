import numpy as np
from scipy.linalg import solve_triangular
import sys

def multivar_norm(x, mu, L):
    ''' L is the Cholesky decomposition of the covaraince '''
    if len(x.shape) == 1:
        x = x.reshape((-1,1))
    if len(mu.shape) == 1:
        mu = mu.reshape((-1,1))
        
    d = x - mu
    alpha = solve_triangular(L, d, lower=True)
    rst =  - 0.5 * x.shape[0] * np.log(2 * np.pi) 
    rst += - np.log(L.diagonal()).sum()
    rst += - 0.5 * (alpha**2).sum()
    return rst
    

def square_dist_multi(X, X2 = None):
    ''' return square distance matrices for each feature '''
    if len(X.shape) == 1:
        X = X.reshape((-1,1))
        if X2 is not None and len(X2.shape) == 1:
            X2 = X2.reshape((-1,1))
            
    if X2 is not None:
        assert (X.shape[1]==X2.shape[1]), "X and X2 must have same number of features"
    
    rst = []
    if X2 is None:
        for kk in range(X.shape[1]):
            rst.append(square_dist(X[:,kk]))
            
    else:
        for kk in range(X.shape[1]):
            rst.append(square_dist(X[:,kk], X2[:,kk]))
            
    return rst


def square_dist(X, X2 = None):
    if len(X.shape) == 1:
        X = X.reshape((-1,1))
        if X2 is not None and len(X2.shape) == 1:
            X2 = X2.reshape((-1,1))
    
    if X2 is not None:
        assert (X.shape[1]==X2.shape[1]), "X and X2 must have same number of features"
    
    Xs = (X**2).sum(1).reshape((-1,1))
    if X2 is None:
        Xs = np.tile(Xs, (1, X.shape[0]))
        return -2*np.dot(X, X.T) + Xs + Xs.T
    else:
        Xs = np.tile(Xs, (1, X2.shape[0]))
        X2s = (X2**2).sum(1).reshape((-1,1))
        X2s = np.tile(X2s, (1, X.shape[0]))
        return -2*np.dot(X, X2.T) + Xs + X2s.T


def kern(dist_mtx, kvar, kll):
    ''' squared exponential kernel '''
    return kvar * np.exp(-0.5*dist_mtx/(kll**2))
    

def kern_multi(dist_mtxs, kvar, klls):
    ''' squared exponential kernel, take a list of dist_mtx and a list of kll '''
    assert (len(dist_mtxs)==len(klls)), "kll number must match the feature number"
    sum = np.zeros_like(dist_mtxs[0])
    for ii, mtx in enumerate(dist_mtxs):
        sum = sum + ( mtx / (klls[ii]**2) )
        
    return kvar * np.exp(-0.5*sum)


def callback(invars):
    ''' callback function for optimization process '''
    kvar, kll, noivar = invars[0:3]
    '''
    print "kvar: %f\n" % kvar
    print "kll**2: %f\n" % kll**2
    print "noivar: %f\n" % noivar
    '''