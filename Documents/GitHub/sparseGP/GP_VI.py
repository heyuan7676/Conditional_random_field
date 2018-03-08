import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from utility_func import multivar_norm, square_dist, kern, callback

class GP_VI(object):
    def __init__(self, X, y, Xm_init, pars):
        self.X = X
        self.y = y
        
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X_dist = square_dist(self.X)
        self.Xm = Xm_init
        self.Xm_ls = [xx for xx in self.Xm.flatten()]
        
        self.kvar = pars[0]
        self.kll = pars[1]
        self.noivar = pars[2]
        self.mllk = self.get_mllk()
        
        print "before training:"
        print "kvar: %f\n" % self.kvar
        print "kll**2: %f\n" % self.kll**2
        print "noivar: %f\n" % self.noivar
        print "-mllk: %f\n" % self.mllk
        print "-approx-mllk: %f\n" % self.obj([self.kvar, self.kll, self.noivar] + self.Xm_ls)
        
        
    def optimize(self):
        bounds = [(1.0e-10, None),(1.0e-10, None),(1.0e-10, None)]
        bounds = bounds + [(None, None)] * len(self.Xm_ls)
        
        result = minimize(self.obj,
                        x0=[self.kvar, self.kll, self.noivar] + self.Xm_ls,
                        method='L-BFGS-B',
                        bounds = bounds,
                        callback=callback,
                        options={'gtol': 1e-6, 'disp': False} )
        
        self.kvar, self.kll, self.noivar = result.x[0:3]
        self.Xm_ls = result.x[3:]
        self.Xm = np.array(self.Xm_ls).reshape((-1, self.d))
        
        self.mllk = self.get_mllk()
        
        self.Xm_dist = square_dist(self.Xm)
        self.Xm_X_dist = square_dist(self.Xm, self.X)
        self.Kuu = kern(self.Xm_dist, self.kvar, self.kll)
        self.Kuf = kern(self.Xm_X_dist, self.kvar, self.kll)
        
        self.L = np.linalg.cholesky(self.Kuu)
        matA = solve_triangular(self.L, self.Kuf, lower=True)/np.sqrt(self.noivar)  # m by n
        matB = np.eye(self.Xm.shape[0]) + np.dot(matA, matA.T)  # m by m
        matBinv = np.linalg.inv(matB)
        self.temp3 = np.eye(self.Xm.shape[0]) - matBinv
        
        Lb = np.linalg.cholesky(matB)  # m by m
        matC = solve_triangular(Lb, np.dot(matA, self.y), lower=True)/np.sqrt(self.noivar) 
        
        temp1 = solve_triangular(Lb.T, matC)
        self.temp2 = solve_triangular(self.L.T, temp1)
        
        print "after training:"
        print "kvar: %f\n" % self.kvar
        print "kll**2: %f\n" % self.kll**2
        print "noivar: %f\n" % self.noivar
        print "-mllk: %f\n" % self.mllk
        print "-approx-mllk: %f\n" % self.obj([self.kvar, self.kll, self.noivar, self.Xm])
        
    
    def predict(self, X_new):
        dist_new = square_dist(X_new, self.Xm)
        K_new = kern(dist_new, self.kvar, self.kll)
        fmean = np.dot(K_new, self.temp2)
        
        dist_new_new = square_dist(X_new)
        K_new_new = kern(dist_new_new, self.kvar, self.kll)
        temp1 = solve_triangular(self.L, K_new.T, lower=True)
        temp4 = solve_triangular(self.L.T, np.dot(self.temp3, temp1))
        fvar = K_new_new - np.dot(K_new, temp4) + np.eye(X_new.shape[0]) * self.noivar
        
        return (fmean, fvar)
    
    
    def get_mllk(self):
        K = kern(self.X_dist, self.kvar, self.kll) + np.eye(self.n) * self.noivar
        try:
            L = np.linalg.cholesky(K)
        except:
            sys.exit("kvar: %f, kll: %f, noivar:%f\n" % (kvar, kll, noivar))
        mu = np.zeros_like(self.y)
        return multivar_norm(self.y, mu, L)
    
    
    def obj(self, invars):
        kvar, kll, noivar = invars[0:3]
        Xm_ls = invars[3:]
        Xm = np.array(Xm_ls).reshape((-1, self.d))
        Xm_dist = square_dist(Xm)
        Xm_X_dist = square_dist(Xm, self.X)
        Kff = kern(self.X_dist, kvar, kll)  # n by n
        Kuu = kern(Xm_dist, kvar, kll) + np.eye(self.Xm.shape[0])*1.0e-10  # m by m
        L = np.linalg.cholesky(Kuu)  # m by m
        Kuf = kern(Xm_X_dist, kvar, kll)  # m by n
        
        matA = solve_triangular(L, Kuf, lower=True)/np.sqrt(noivar)  # m by n
        matB = np.eye(Xm.shape[0]) + np.dot(matA, matA.T)  # m by m
        
        Lb = np.linalg.cholesky(matB)  # m by m
        matC = solve_triangular(Lb, np.dot(matA, self.y), lower=True)/np.sqrt(noivar)  # m by 1
        rst =  - 0.5 * self.n * np.log(2 * np.pi)
        rst += - 0.5 * np.log(np.linalg.det(matB))
        rst += - 0.5 * self.n * np.log(noivar)
        rst += - 0.5 * np.dot(self.y.T, self.y)/noivar
        rst += 0.5 * np.dot(matC.T, matC)
        rst += -0.5 * np.trace(Kff)/noivar
        rst += 0.5 * np.trace(np.dot(matA, matA.T))
        
        return -rst
    
