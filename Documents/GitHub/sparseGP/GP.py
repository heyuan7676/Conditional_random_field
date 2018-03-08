import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from utility_func import multivar_norm, square_dist, kern, callback

class GP(object):
    def __init__(self, X, y, pars):
        self.X = X
        self.y = y
        
        self.n = X.shape[0]
        self.X_dist = square_dist(self.X)
        
        self.kvar = pars[0]
        self.kll = pars[1]
        self.noivar = pars[2]
        self.mllk = self.obj([self.kvar, self.kll, self.noivar])
        
        print "before training:"
        print "kvar: %f\n" % self.kvar
        print "kll**2: %f\n" % self.kll**2
        print "noivar: %f\n" % self.noivar
        print "mllk: %f\n" % self.mllk
        
        
    def optimize(self):
        result = minimize(self.obj,
                        x0=[self.kvar, self.kll, self.noivar],
                        method='L-BFGS-B',
                        bounds = ((1.0e-10, None),(1.0e-10, None),(1.0e-10, None)),
                        callback=callback,
                        options={'gtol': 1e-6, 'disp': False} )
        
        self.kvar, self.kll, self.noivar = result.x
        self.mllk = self.obj([self.kvar, self.kll, self.noivar])
        self.KX = kern(self.X_dist, self.kvar, self.kll) + np.eye(self.n) * self.noivar
        self.L = np.linalg.cholesky(self.KX)
        self.V = solve_triangular(self.L, self.y, lower=True)
        
        print "after training:"
        print "kvar: %f\n" % self.kvar
        print "kll**2: %f\n" % self.kll**2
        print "noivar: %f\n" % self.noivar
        print "mllk: %f\n" % self.mllk
        
    
    def predict(self, X_new):
        dist_new = square_dist(self.X, X_new)
        K_new = kern(dist_new, self.kvar, self.kll)
        A = solve_triangular(self.L, K_new, lower=True)
        fmean = np.dot(A.T, self.V)
        
        dist_new_new = square_dist(X_new)
        K_new_new = kern(dist_new_new, self.kvar, self.kll)
        fvar = K_new_new - np.dot(A.T, A) + np.eye(X_new.shape[0]) * self.noivar
        return (fmean, fvar)
    
    
    def obj(self, invars):
        kvar, kll, noivar = invars
        K = kern(self.X_dist, kvar, kll) + np.eye(self.n) * noivar
        try:
            L = np.linalg.cholesky(K)
        except:
            sys.exit("kvar: %f, kll: %f, noivar:%f\n" % (kvar, kll, noivar))
        mu = np.zeros_like(self.y)
        return -multivar_norm(self.y, mu, L)
        
    
    def obj_der(self, invars):
        print "called obj_der"
        kvar, kll, noivar = invars
        K = kern(self.X_dist, kvar, kll) + np.eye(self.n) * noivar
        L = np.linalg.cholesky(K)
        alpha = solve_triangular(L, self.y, lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        L_inv = solve_triangular(L, np.eye(self.n), lower=True)
        K_inv = np.dot(L_inv.T, L_inv)
        term1 = np.dot(alpha, alpha.T) - K_inv
        return np.array([self.obj_der_kvar(kll, term1),
                self.obj_der_kll(kvar, kll, term1),
                self.obj_der_noivar(term1)])
    
    
    def obj_der_kvar(self, kll, term1):
        term2 = kern(self.X_dist, 1.0, kll)
        return -0.5*np.trace(np.dot(term1,term2))
    
    
    def obj_der_kll(self, kvar, kll, term1):
        term2 = kvar * np.exp(-self.X_dist/(2.0*(kll**2))) * self.X_dist / (kll**3)
        return -0.5*np.trace(np.dot(term1,term2))
    
    
    def obj_der_noivar(self, term1):
        return -0.5*np.trace(term1)
    
