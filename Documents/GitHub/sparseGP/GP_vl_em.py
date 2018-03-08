import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
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


def square_dist(X, X2 = None):
    if len(X.shape) == 1:
        X = X.reshape((-1,1))
    
    Xs = (X**2).sum(1).reshape((-1,1))
    if X2 is None:
        Xs = np.tile(Xs, (1, X.shape[0]))
        return -2*np.dot(X, X.T) + Xs + Xs.T
    else:
        if len(X2.shape) == 1:
            X2 = X2.reshape((-1,1))
            
        Xs = np.tile(Xs, (1, X2.shape[0]))
        X2s = (X2**2).sum(1).reshape((-1,1))
        X2s = np.tile(X2s, (1, X.shape[0]))
        return -2*np.dot(X, X2.T) + Xs + X2s.T


def kern(dist_mtx, kvar, kll):
    return kvar * np.exp(-0.5*dist_mtx/(kll**2))


def callback(invars):
    kvar, kll, noivar = invars
    print "kvar: %f\n" % kvar
    print "kll: %f\n" % kll
    print "noivar: %f\n" % noivar
    

class GP(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        self.n = X.shape[0]
        self.X_dist = square_dist(self.X)
        
        self.kvar = 1.0
        self.kll = 1.0
        self.noivar = 1.0
        self.mllk = self.obj([self.kvar, self.kll, self.noivar])
        
        print "kvar: %f\n" % self.kvar
        print "kll: %f\n" % self.kll
        print "noivar: %f\n" % self.noivar
        print "mllk: %f\n" % self.mllk
        
        
    def optimize(self):
        
        for i in range(self.num_Xm):
            print "iter: %d\n" % i
            # e-step
            negmllk_ls = []
            for j in range(self.n - i):
                # try add xj into Xm
                curr_xx = self.X_ex_Xm[j, :].reshape((1,-1))
                self.X_ex_Xm = np.delete(self.X_ex_Xm, j, 0)
                self.Xm = np.append(self.Xm, curr_xx, axis=0)
                
                negmllk_ls.append(negmllk_ls, self.obj([self.kvar, self.kll, self.noivar]))
                
                # put xj back
                self.Xm = np.delete(self.Xm, self.Xm.shape[0]-1, 0)
                self.X_ex_Xm = np.insert(self.X_ex_Xm, j, curr_xx, 0)
                
            # greedily choose the best xx to add to Xm
            id_min = np.argmin(negmllk_ls)
            x_min = self.X_ex_Xm[id_min, :].reshape((1,-1))
            self.X_ex_Xm = np.delete(self.X_ex_Xm, id_min, 0)
            self.Xm = np.append(self.Xm, x_min, axis=0)
            
            
            # m-step
            result = minimize(self.obj,
                     x0=[self.kvar, self.kll, self.noivar],
                     method='L-BFGS-B',
                     bounds = ((1.0e-10, None),(1.0e-10, None),(1.0e-10, None)),
                     callback=callback,
                     options={'gtol': 1e-6, 'disp': True} )
                     
            self.kvar, self.kll, self.noivar = result.x
            self.mllk = self.obj([self.kvar, self.kll, self.noivar])
        
        
        
        print "kvar: %f\n" % self.kvar
        print "kll: %f\n" % self.kll
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
        print "called obj"
        kvar, kll, noivar = invars
        Kff = kern(self.X_dist, kvar, kll)  # n by n
        Xm_dist = square_dist(self.Xm)
        Kuu = kern(Xm_dist, kvar, kll)  # m by m
        L = np.linalg.cholesky(Kuu)  # m by m
        Xm_X dist = square_dist(self.Xm, self.X)
        Kuf = kern(Xm_X_dist, kvar, kll)  # m by n
        
        matA = solve_triangular(L, Kuf, lower=True)/np.sqrt(noivar)  # m by n
        matB = np.eye() + np.dot(matA, matA.T)  # m by m
        
        Lb = np.linalg.cholesky(matB)  # m by m
        matC = solve_triangular(Lb, np.dot(matA, self.y), lower=True)/np.sqrt(noivar)  # m by 1
        rst =  - 0.5 * self.n * np.log(2 * np.pi)
        rst += - 0.5 * np.log(np.linalg.det(matB))
        rst += - 0.5 * self.n * np.log(noivar)
        rst += - 0.5 * np.dot(self.y.T, self.y)/noivar
        rst += 0.5 * np.dot(matC.T, matC)
    
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
    
    
def main():
    X = np.loadtxt('./SPGP_dist/train_inputs').reshape((-1,1))
    y = np.loadtxt('./SPGP_dist/train_outputs').reshape((-1,1))
    myGP = GP(X, y)
    myGP.optimize()
    
    xx = np.linspace(-4,9,100).reshape(-1,1)
    yy, yy_var = myGP.predict(xx)
    yy_var = (np.diagonal(yy_var)**0.5).reshape(-1,1)
    
    plt.close()
    plt.plot(X, y, 'x')
    plt.plot(xx, yy)
    
    plt.plot(xx, yy + 2.0*yy_var)
    plt.plot(xx, yy - 2.0*yy_var)
    
    plt.savefig('test.png')
    
    
if __name__ == "__main__":
    main()
