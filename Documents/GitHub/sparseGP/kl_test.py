import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from GP import GP
from GP_VI import GP_VI
from GP_PP import GP_PP

def gau_kl(pm, pv, qm, qv):
    '''
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    pv and qv are covariance matrices, n by n
    pm and qm are also n by 1
    '''
    dpv = np.linalg.det(pv)
    dqv = np.linalg.det(qv)
    iqv = np.linalg.inv(qv)
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + np.trace(iqv * pv)         # + tr(\Sigma_q^{-1} * \Sigma_p)
             + reduce(np.dot, [diff.T, iqv, diff])   # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))

def run_GP(X, y, xx, pars, model='full', induce_num = 15, get_par = False):
    _start = time.time()
    if model == 'full':
        myGP = GP(X, y, pars)
    else:
        n = X.shape[0]
        m = np.random.choice(range(n), induce_num, replace=False)
        Xm_init = X[m, :]
        if model == 'vi':
            myGP = GP_VI(X, y, Xm_init, pars)
        elif model == 'pp':
            myGP = GP_PP(X, y, Xm_init, pars)
        
    myGP.optimize()
    _traintime = time.time()
    yy, yy_var = myGP.predict(xx)
    _predicttime = time.time()
    
    print "model %s training time: %f" % (model, _traintime - _start)
    print "model %s predicting time: %f" % (model, _predicttime - _traintime)
    
    if get_par == False:
        return (yy, yy_var)
    else:
        return (yy, yy_var, (myGP.kvar, myGP.kll, myGP.noivar))


def main():
    X = np.loadtxt('./data/housing_train_x.txt')
    y = np.loadtxt('./data/housing_train_y.txt').reshape((-1,1))
    if len(X.shape) == 1:
        X = X.reshape((-1,1))
        
    xx = np.loadtxt('./data/housing_test_x.txt')
    yy_t = np.loadtxt('./data/housing_test_y.txt').reshape((-1,1))
    
    pars = [0.001, 0.5, 1.0]
    yy, yy_var, pars = run_GP(X, y, xx, pars, 'full', 15, True)
    print "*** FULL SSE: %f\n" % ((yy-yy_t)**2).sum()
    
    
    '''
    tests_idn = np.linspace(20,300,15)
    kl1 = []
    kl2 = []
    for induce_num in tests_idn:
        yy1, yy_var1 = run_GP(X, y, xx, pars, 'vi', int(induce_num))
        yy2, yy_var2 = run_GP(X, y, xx, pars, 'pp', int(induce_num))
        kl1.append(gau_kl(yy, yy_var, yy1, yy_var1))
        kl2.append(gau_kl(yy, yy_var, yy2, yy_var2))
        print "*** VI SSE: %f\n" % ((yy1-yy_t)**2).sum()
        print "*** PP SSE: %f\n" % ((yy2-yy_t)**2).sum()
    
    with open('housing_kl.txt','w') as fh:
        for ii in range(len(tests_idn)):
            fh.write("%d\t%f\t%f\n" % (int(tests_idn[ii]), kl1[ii], kl2[ii]))
    '''
    
    '''
    plt.close()
    plt.plot(tests_idn, kl1, label='VI GP')
    plt.plot(tests_idn, kl2, label='PP GP')
    plt.plot(tests_idn, [0]*len(tests_idn), 'b--')
    plt.xlabel('Number of inducing variables')
    plt.ylabel('KL(p||q)')
    plt.legend()
    plt.savefig('housing_kl.png')
    '''
    
if __name__ == "__main__":
    main()