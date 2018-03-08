import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from utility_func import multivar_norm, square_dist, kern, callback
from GP_VI import GP_VI


class GP_PP(GP_VI):
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
        
        return -rst
    
