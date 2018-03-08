import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from utility_func import multivar_norm, square_dist_multi, kern_multi, callback

class GP(object):
    def __init__(self, X, y, pars):
        self.X = X
        self.y = y
        
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X_dists = square_dist_multi(self.X)
        assert (len(self.X_dists) == self.d), "self.X_dists length is not right"
        
        self.kvar = pars[0]
        self.klls = pars[1]
        assert (len(self.klls) == self.d), "input klls length is not right"
        self.noivar = pars[2]
        self.mllk = self.obj([self.kvar] + self.klls + [self.noivar])
        
        print "before training:"
        print "kvar: %f\n" % self.kvar
        print "klls**2: %s\n" % [str(xx**2) for xx in self.klls]
        print "noivar: %f\n" % self.noivar
        print "mllk: %f\n" % self.mllk
        
        
    def optimize(self):
        bounds = [(1.0e-10, None)] * (len(self.klls)+2)
        
        result = minimize(self.obj,
                        x0=[self.kvar] + self.klls + [self.noivar],
                        method='L-BFGS-B',
                        bounds = bounds,
                        callback=callback,
                        options={'gtol': 1e-6, 'disp': False} )
        
        print "train is done"
        print result.x
        
        self.kvar = result.x[0]
        self.klls = result.x[1:-1]
        assert (len(self.klls) == self.d), "after op, klls length is not right"
        self.noivar = result.x[-1]
        
        print "after training:"
        print "kvar: %f\n" % self.kvar
        print "klls**2: %s\n" % [str(xx**2) for xx in self.klls]
        print "noivar: %f\n" % self.noivar
        
        self.mllk = self.obj(result.x)
        self.KX = kern_multi(self.X_dists, self.kvar, self.klls) + np.eye(self.n) * self.noivar
        self.L = np.linalg.cholesky(self.KX)
        self.V = solve_triangular(self.L, self.y, lower=True)
        
        
        print "mllk: %f\n" % self.mllk
        
    
    def predict(self, X_new):
        assert (self.d == X_new.shape[1]), "new input Xs feature number is not right"
        dist_new = square_dist_multi(self.X, X_new)
        K_new = kern_multi(dist_new, self.kvar, self.klls)
        A = solve_triangular(self.L, K_new, lower=True)
        fmean = np.dot(A.T, self.V)
        
        dist_new_new = square_dist_multi(X_new)
        K_new_new = kern_multi(dist_new_new, self.kvar, self.klls)
        fvar = K_new_new - np.dot(A.T, A) + np.eye(X_new.shape[0]) * self.noivar
        return (fmean, fvar)
    
    
    def obj(self, invars):
        kvar = invars[0]
        klls = invars[1:-1]
        noivar = invars[-1]
        assert (len(klls) == self.d), "in obj klls lengths is not right %d" % len(klls)
        
        K = kern_multi(self.X_dists, kvar, klls) + np.eye(self.n) * noivar
        try:
            L = np.linalg.cholesky(K)
        except:
            sys.exit("kvar: %f, kll: %f, noivar:%f\n" % (kvar, klls, noivar))
        mu = np.zeros_like(self.y)
        return -multivar_norm(self.y, mu, L)
    
    
