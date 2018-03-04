import argparse
import pdb

import pandas as pd
import numpy as np
import os
import sys
from itertools import combinations

from sklearn import metrics
from scipy.linalg.blas import dgemm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pickle
import time


from VI_MF import VI_MF_Cython




#############################################################################################
##
##   Run CRF model to compute potentials on pairwise edges
##
#############################################################################################



def sigmoid(a):
	return 1 / (1 + np.exp (-a))



def objective_function_outside(self, updating_pairs, update_Ypos):
        # obj = log[P(y,X)] - log[Z(X)]

        # true distribution
        changed_Y = self.Y[updating_pairs]
        potential_YY = np.sum(np.sum(changed_Y[np.newaxis].T * changed_Y[np.newaxis],1) * self.lmda / 2)
        potential_YX = np.sum(np.dot(self.X[:,updating_pairs,:] * self.Y[updating_pairs][np.newaxis].T, self.w))
        l2norm = 2 * (self.alpha1 * np.sum(np.power(self.lmda, 2)) / 2 + self.alpha0 * np.sum(np.power(self.w, 2)))
        joint_llk = potential_YY + potential_YX - l2norm

        # under posterior distribution
	if update_Ypos:
		self.VI_E_y(updating_pairs)
        changed_Y = self.E_Y_given_theta[updating_pairs]
        potential_YY = np.sum(np.sum(changed_Y[np.newaxis].T * changed_Y[np.newaxis],1) * self.lmda / 2)
        potential_YX = np.sum(np.dot(self.X[:,updating_pairs,:] * self.E_Y_given_theta[updating_pairs][np.newaxis].T, self.w))
        normalization_term = potential_YY + potential_YX

        obj = joint_llk - normalization_term
	return obj


class CRF_object():
    
    def __init__(self, tissues, features, train_Y, train_X, test_Y, test_X, elbo_episoron, obj_episoron, max_iter, general_max_iter, eta, alpha0, alpha1):
         ##data
        self.tissues = tissues
        self.features = features
        self.T = len(self.tissues)
        self.N = len(train_Y)
        self.O = train_X[tissues[0]].shape[1]
        
        self.Y = train_Y
        self.X = []
        for t in xrange(self.T):
            self.X.append(train_X[self.tissues[t]])
        self.X = np.array(self.X)
        
        self.test_Y = test_Y
        self.test_X = []
        for t in xrange(self.T):
            self.test_X.append(test_X[self.tissues[t]])
        self.test_X = np.array(self.test_X)
        
        ## super parameters
        self.elbo_episoron = elbo_episoron
        self.obj_episoron = obj_episoron
        self.max_iter = max_iter
        self.general_max_iter = general_max_iter
	self.eta_w = eta
	self.eta_lmda = eta
	self.alpha0 = alpha0
	self.alpha1 = alpha1
        
    def initialize(self):
	# initialize potential between yt - xo edges
        self.w = np.random.sample(self.O)
	#self.w = (self.w - 0.5) * 2
        # initialize potential between ys - yt edges
        self.lmda = np.random.sample([self.T, self.T])
        self.lmda = (self.lmda + self.lmda.T) / 2   ## symmetric!
	#self.lmda = (self.lmda - 0.5) * 2
	np.fill_diagonal(self.lmda, 0)
        # posterior distribution of Y
        self.E_Y_given_theta = np.random.sample([self.N, self.T])
        # parameters to store
        self.pairs_batch = []
  

    def VI_E_y(self, updating_pairs):
	self.compute_ELBO(self.E_Y_given_theta[updating_pairs], self.X[:, updating_pairs,: ])
        self.old_ELBO = self.ELBO.copy()

        self.E_Y_given_theta[updating_pairs] = sigmoid(np.dot(self.lmda, self.E_Y_given_theta[updating_pairs].T) + np.dot(self.X[:,updating_pairs,:], self.w)).T
        self.compute_ELBO(self.E_Y_given_theta[updating_pairs], self.X[:, updating_pairs,: ])
        self.ELBO_change = [self.ELBO - self.old_ELBO]

        self.old_ELBO = self.ELBO
        iteration = 1

        while (self.ELBO_change[-1] >= self.elbo_episoron ) and (iteration < self.max_iter):
	    #a = sigmoid(np.dot(self.lmda, self.E_Y_given_theta[updating_pairs].T) + np.dot(self.X[:,updating_pairs,:], self.w)).T
            b = sigmoid(np.asarray(VI_MF_Cython(self.X[:,updating_pairs,:], self.E_Y_given_theta[updating_pairs], self.lmda, self.w, len(updating_pairs), self.T)))
	    self.E_Y_given_theta[updating_pairs] = b
            #print 'Difference', np.sum(b.ravel()-a.ravel())

            self.compute_ELBO(self.E_Y_given_theta[updating_pairs], self.X[:, updating_pairs,: ])
	    self.ELBO_change.append(self.ELBO - self.old_ELBO)
	    self.old_ELBO = self.ELBO
	    iteration += 1

        if iteration == self.max_iter:
            print 'VI: Maximum iteration reached!'
        else:
            pass

	if np.sum(np.array(self.ELBO_change) < 0) > 0:
		if np.abs(self.ELBO_change[-1] - -7.27595761418e-12) < 1e-12:
			pdb.set_trace()
		## no idea why!!!
		print 'VI update caused negative ELBO', self.ELBO_change[-1]


    def compute_ELBO(self, changed_Y, involved_X):
        # using VI, log[Z(X)] = Eq[logP(y)] - Eq[logq(y)]
        potential_YY = np.sum(np.sum(changed_Y[np.newaxis].T * changed_Y[np.newaxis],1) * self.lmda / 2)
        potential_YX = np.sum(np.dot(involved_X * changed_Y[np.newaxis].T, self.w))
        Eq_logp = potential_YY + potential_YX
        Eq_logq = np.sum(changed_Y * np.log(changed_Y + 1e-10) + (1-changed_Y) * np.log(1-changed_Y + 1e-10))
        self.ELBO = Eq_logp - Eq_logq


    def objective_function(self, updating_pairs, update_Ypos):
        # obj = log[P(y,X)] - log[Z(X)]
        
        # true distribution
        changed_Y = self.Y[updating_pairs]
        potential_YY = np.sum(np.sum(changed_Y[np.newaxis].T * changed_Y[np.newaxis],1) * self.lmda / 2)
        potential_YX = np.sum(np.dot(self.X[:,updating_pairs,:] * self.Y[updating_pairs][np.newaxis].T, self.w))
	l2norm = 2 * (self.alpha1 * np.sum(np.power(self.lmda, 2)) / 2 + self.alpha0 * np.sum(np.power(self.w, 2)))
        joint_llk = potential_YY + potential_YX - l2norm
        
        # under posterior distribution
	if update_Ypos:
		self.VI_E_y(updating_pairs)
        changed_Y = self.E_Y_given_theta[updating_pairs]
        potential_YY = np.sum(np.sum(changed_Y[np.newaxis].T * changed_Y[np.newaxis],1) * self.lmda / 2)
        potential_YX = np.sum(np.dot(self.X[:,updating_pairs,:] * self.E_Y_given_theta[updating_pairs][np.newaxis].T, self.w))
        normalization_term = potential_YY + potential_YX

        self.obj = joint_llk - normalization_term



    def gradient_ascent_lmda(self, updating_pairs, iteration):

	self.VI_E_y(updating_pairs)
        lmda_gradient = np.mean(self.Y[updating_pairs][np.newaxis].T * self.Y[updating_pairs][np.newaxis] - self.E_Y_given_theta[updating_pairs][np.newaxis].T * self.E_Y_given_theta[updating_pairs][np.newaxis], axis=1) - 2 * self.alpha1 * self.lmda
        self.lmda_G2.append(np.power(lmda_gradient,2))
        if iteration <= 1:
                lmda_Eg2 = (1-gamma) * np.array(self.lmda_G2[-1])
        else:
                lmda_Eg2 = gamma * np.mean(self.lmda_G2[:-1], axis=0) + (1-gamma) * np.array(self.lmda_G2[-1])
        lmda_RMS_g = np.sqrt(lmda_Eg2 + 1e-8)

        #if iteration <= 1:
        #       lmda_EdeltaTheta2 = (1-gamma) * np.array(self.lmda_deltaTheta2[0])
        #else:
        #       lmda_EdeltaTheta2 = gamma * np.mean(self.lmda_deltaTheta2[:-1], axis=0) + (1-gamma) * np.array(self.lmda_deltaTheta2[-1])
        #lmda_RMS_delta_theta = np.sqrt(lmda_EdeltaTheta2 + 1e-8)
        #lmda_deltaTheta_t = lmda_RMS_delta_theta / lmda_RMS_g * lmda_gradient

        lmda_deltaTheta_t = self.eta_lmda / lmda_RMS_g * lmda_gradient
        self.lmda = self.lmda + lmda_deltaTheta_t

        #self.lmda_deltaTheta2.append(np.power(lmda_deltaTheta_t,2))

        np.fill_diagonal(self.lmda, 0)


    def gradient_ascent_w(self, updating_pairs, iteration):

	self.VI_E_y(updating_pairs)
        w_gradient = np.mean(np.mean((self.Y[updating_pairs,:] - self.E_Y_given_theta[updating_pairs,:])[np.newaxis].T * self.X[:,updating_pairs,:], axis=0), axis=0) - 2 * self.alpha0 * self.w
        self.G2.append(np.power(w_gradient,2))
        if iteration <= 1:
                Eg2 = (1-gamma) * np.array(self.G2[0])
        else:
                Eg2 = gamma * np.mean(self.G2[:-1], axis=0) + (1-gamma) * np.array(self.G2[-1])
        RMS_g = np.sqrt(Eg2 + 1e-8)

        #if iteration <= 1:
        #       EdeltaTheta2 = (1-gamma) * np.array(self.deltaTheta2[0])
        #else:
        #       EdeltaTheta2 = gamma * np.mean(self.deltaTheta2[:-1], axis=0) + (1-gamma) * np.array(self.deltaTheta2[-1])
        #RMS_delta_theta = np.sqrt(EdeltaTheta2 + 1e-8)
        #deltaTheta_t = RMS_delta_theta / RMS_g * w_gradient

        deltaTheta_t = self.eta_w / RMS_g * w_gradient
        self.w = self.w + deltaTheta_t

        #self.deltaTheta2.append(np.power(deltaTheta_t,2))


    def update_lmda(self, pairs_in_batch, iteration):
	old_obj = self.obj
	self.old_lmda = self.lmda.copy()
	self.old_E_Y_given_theta = self.E_Y_given_theta[pairs_in_batch,:].copy()
	flag = 0

	self.gradient_ascent_lmda(pairs_in_batch, iteration)
	obj = objective_function_outside(self, pairs_in_batch, update_Ypos = 1)

	while (obj - old_obj < 0) and (self.eta_lmda > 1e-50):
		self.eta_lmda = self.eta_lmda / 5
		self.lmda = self.old_lmda.copy()
		self.E_Y_given_theta[pairs_in_batch,:] = self.old_E_Y_given_theta.copy()
		self.gradient_ascent_lmda(pairs_in_batch, iteration)
		obj = objective_function_outside(self, pairs_in_batch, update_Ypos = 0)
		flag = 1

	if flag:
		print 'Objective decreases! Lmda learning rate is too large!! (iteration %d)  --> decrease eta (%f)!' % (iteration, self.eta_lmda)

	if obj - old_obj < 0:
		self.lmda = self.old_lmda.copy()
		self.E_Y_given_theta[pairs_in_batch,:] = self.old_E_Y_given_theta




    def update_w(self, pairs_in_batch, iteration):
        old_obj = self.obj
        self.old_w = self.w.copy()
	self.old_E_Y_given_theta = self.E_Y_given_theta[pairs_in_batch,:].copy()
	flag = 0

        self.gradient_ascent_w(pairs_in_batch, iteration)
	obj = objective_function_outside(self, pairs_in_batch, update_Ypos = 1)

        while (obj - old_obj < 0) and (self.eta_w > 1e-50):
                self.eta_w = self.eta_w / 5
                self.w = self.old_w.copy()
		self.E_Y_given_theta[pairs_in_batch,:] = self.old_E_Y_given_theta.copy()
                self.gradient_ascent_w(pairs_in_batch, iteration)
		obj = objective_function_outside(self, pairs_in_batch, update_Ypos = 0)
		flag = 1

	if flag:
		print 'Objective decreases! W learning rate is too large!! (iteration %d)  --> decrease eta (%f)!' % (iteration, self.eta_w)

	if obj - old_obj < 0:
		self.w = self.old_w.copy()
		self.E_Y_given_theta[pairs_in_batch,:] = self.old_E_Y_given_theta
	


    def update_batch(self, pairs_in_batch):

        # 2) Update E(Y|X,Theta) on these data points
        #self.VI_E_y(pairs_in_batch)

      
        # 1) compute the old objective 
        self.objective_function(pairs_in_batch, update_Ypos = 1)
        self.old_obj = self.obj

 
        # 3) first updating
        iteration = 1
        self.G2 = []
        self.deltaTheta2 = [np.ones(self.O) * self.eta_w]
        self.lmda_G2 = []
        self.lmda_deltaTheta2 = [np.ones([self.T,self.T]) * self.eta_lmda]

	self.update_lmda(pairs_in_batch, iteration)
	self.update_w(pairs_in_batch, iteration)

        self.objective_function(pairs_in_batch, update_Ypos = 0)
        self.obj_change = [self.obj - self.old_obj]


	while ((self.obj_change[-1] >= self.obj_episoron) or (self.obj_change[-1] < -self.obj_episoron)) and (iteration < self.max_iter): 

            # 1) save the old objective
            self.old_obj = self.obj

            # 2) compute posterior E(y), used by gradient ascent
            #self.VI_E_y(pairs_in_batch)
            
	    # 3) update
	    self.update_lmda(pairs_in_batch, iteration)
            self.update_w(pairs_in_batch, iteration)

	    self.objective_function(pairs_in_batch, update_Ypos = 0)
            self.obj_change.append(self.obj - self.old_obj)

            iteration += 1
	    #print self.obj_change[-1]

	    if  (self.obj_change[-1] < 0):
		self.lmda = self.old_lmda.copy()
		self.w = self.old_w.copy()
		self.E_Y_given_theta[pairs_in_batch,:] = self.old_E_Y_given_theta
		break

	print 'Final eta:', self.eta_lmda, self.eta_w

        # test the AUC for pairs in the batch
        [fpr, tpr, _]= metrics.roc_curve(self.Y[pairs_in_batch].ravel(), self.E_Y_given_theta[pairs_in_batch].ravel())
        train_AUC = metrics.auc(fpr, tpr)

	print 'Last 10 obj_change', self.obj_change[-10:]
	if iteration == self.max_iter:
        	print 'Running CRF (With TFs, shared w): training AUC for this batch = %2.3f; ' % (train_AUC), 'PERMUTE = ', PERMUTE, 'Maximum iteration reached!'
	else:
		print 'Running CRF (With TFs, shared w): training AUC for this batch = %2.3f; ' % (train_AUC), 'PERMUTE = ', PERMUTE, 'Converged after %d iterations' % iteration

	# if train_AUC < 0.5:
	# 	pdb.set_trace()


	self.eta_w = eta
	self.eta_lmda = eta


    def update(self):
        
        #### update parameters
        # 1) choose the data points to update
        pairs_in_batch = np.random.choice(range(self.N), int(self.N * batch), replace=False)
        self.pairs_batch.append(pairs_in_batch)
        
        # 1.1) compute the old objective
	print 'Initial obj'
        self.objective_function(range(self.N), update_Ypos = 1)
        self.old_full_obj = self.obj
       
 
        # 2) update using these data points
        self.update_batch(pairs_in_batch)

        # 2.2) compute the updated objective
        self.objective_function(range(self.N), update_Ypos = 1)
        self.entire_obj_change = [self.obj - self.old_full_obj]
        iteration = 1
	self.old_full_obj = self.obj

        while ( np.abs((self.entire_obj_change[-1])) >= self.obj_episoron) and (iteration < self.general_max_iter):
            # update in batches
            pairs_in_batch = np.random.choice(range(self.N), int(self.N * batch), replace=False)
            self.pairs_batch.append(pairs_in_batch)
            self.update_batch(pairs_in_batch)
            iteration += 1
            
            # check obj change for the full set of data
            self.objective_function(range(self.N), update_Ypos = 1)
            self.entire_obj_change.append(self.obj - self.old_full_obj)
            self.old_full_obj = self.obj


        if iteration == self.general_max_iter:
            print 'Model: Maximum iteration reached!'
        else:
            print 'Model: Converged in %d iterations' % iteration
                
        [fpr, tpr, _]= metrics.roc_curve(self.Y.ravel(), self.E_Y_given_theta.ravel())
        train_AUC = metrics.auc(fpr, tpr)
        print
        print 'Model setting:'
        print 'W shared across tissues;'
        print 'There are %d training data points and %d test data points ' % (len(self.Y), len(self.test_Y))
        print 'Mini-batch update, each batch has %i data points, overall %d data points are used in training' % (int(len(self.Y) * batch), len(np.unique(np.array(self.pairs_batch).ravel())))
        print 'Trained on %d features' % self.O
	print 'Initialial learning rate %f' % eta
        print 'Parameters: mvalue threshold = %s; alpha0 = %s, alpha1 = %s' % (str(mvalue_threshold), str(self.alpha0), str(self.alpha1))
        print 'Stop criterion: obj_episoron = %s, max_iter = %s, general_max_iter = %s' % (str(self.obj_episoron), str(self.max_iter), str(self.general_max_iter))
        print 'Running CRF (With TFs, sharing w): overall training AUC=%2.3f; ' % train_AUC, 'PERMUTE = ', PERMUTE
	print 'Entire objective change increase:', np.sum(self.entire_obj_change)


    def inference(self):
        # substitue the data
        self.X = self.test_X
        self.N = len(self.X[0])
        
        # initialize
        self.E_Y_given_theta = np.random.sample([self.N, self.T])
        self.compute_ELBO(self.test_Y, self.X)
        self.old_ELBO = self.ELBO
        
        # update E(y|X, theta) until convergence
        self.VI_E_y(range(self.N))
        
        [fpr, tpr, _]= metrics.roc_curve(self.test_Y.ravel(), self.E_Y_given_theta.ravel())
        test_AUC = metrics.auc(fpr, tpr)
        print 'Running CRF: test AUC=%2.3f; ' % test_AUC, 'PERMUTE = ', PERMUTE



def main():
	inputfn = '%s/results/CRF_dataset_%s_%s_%s_ChIP_TFs' % (datadir, str(mvalue_threshold), str(gene_exp_threshold), str(gene_sd_threshold))

	if compute_dataset:	
		tissues = ['Adipose_Subcutaneous', 'Adrenal_Gland', 'Artery_Aorta', 'Artery_Coronary', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 'Brain_Putamen_basal_ganglia', 'Breast_Mammary_Tissue', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Mucosa', 'Heart_Left_Ventricle', 'Muscle_Skeletal', 'Ovary', 'Pancreas', 'Prostate', 'Spleen', 'Stomach', 'Testis', 'Uterus']
	        tissues = ['Adipose_Subcutaneous', 'Artery_Aorta', 'Colon_Sigmoid', 'Muscle_Skeletal', 'Stomach']
		[train_Y, train_X, test_Y, test_X, train_pairs, test_pairs, features] = construct_datasets(tissues)
		np.savez(inputfn, train_Y = train_Y, train_X = train_X, test_Y = test_Y, test_X = test_X, train_pairs = train_pairs, test_pairs = test_pairs, features = features, tissues = tissues)
		
	data = np.load('%s.npz' % inputfn)
	train_Y = data['train_Y']
	train_X = data['train_X'].item()
	test_Y = data['test_Y']
	test_X = data['test_X'].item()
	train_pairs = data['train_pairs']
	test_pairs = data['test_pairs']
	features = data['features']
	tissues = data['tissues']

	#feature_to_use_idx = [list(data['features']).index(selected_features[x]) for x in range(len(selected_features))]
	#features = features[feature_to_use_idx]
	#for k in train_X.keys():
#		train_X[k] = train_X[k][:,feature_to_use_idx]
	#for k in test_X.keys():
	#	test_X[k] = test_X[k][:,feature_to_use_idx]

	if PERMUTE:
		train_Y = np.random.permutation(train_Y)
		test_Y = np.random.permutation(test_Y)
                #for k in train_X.keys():
		#	train_X[k] = np.array(map(lambda t: np.random.permutation(t), train_X[k].T)).T
                #for k in test_X.keys():
                #        test_X[k] = np.array(map(lambda t: np.random.permutation(t), test_X[k].T)).T


	OBJECT = CRF_object(tissues, features, train_Y, train_X, test_Y, test_X, elbo_episoron, obj_episoron, max_iter, general_max_iter, eta, alpha0, alpha1)
	OBJECT.initialize()

	print 'Train the model'
	S = time.time()
	OBJECT.update()
	print 'Training takes ', (time.time()-S) / 60 / 60, 'h'

	print 'Do inference on test data'
	OBJECT.inference()
	print 
	print
	print

	if PERMUTE:
                outfile = '%s/results/CRF_OBJECT_%s_%s_%s_batch%s_eta%s_maxiter%s_generalmaxiter%s_adagrad_speedup_l2Norm_alpha0%s_alpha1%s_ChIP_allLmda_permute.p' % (datadir, str(mvalue_threshold), str(gene_exp_threshold), str(gene_sd_threshold), str(batch), str(eta), str(max_iter), str(general_max_iter), str(alpha0), str(alpha1))
        else:
                outfile = '%s/results/CRF_OBJECT_%s_%s_%s_batch%s_eta%s_maxiter%s_generalmaxiter%s_adagrad_speedup_l2Norm_alpha0%s_alpha1%s_ChIP_allLmda.p' % (datadir, str(mvalue_threshold), str(gene_exp_threshold), str(gene_sd_threshold), str(batch), str(eta), str(max_iter), str(general_max_iter), str(alpha0), str(alpha1))
	with open(outfile, 'wb') as fn:
		pickle.dump(OBJECT, fn)

     
 
if __name__ == '__main__':

    datadir = '/scratch1/battle-fs1/heyuan/tissue_spec_eQTL/data'

    parser = argparse.ArgumentParser(description = 'Learn CRF model')
    parser.add_argument('-D', '--Dataset', dest = 'compute_dataset', default = 0, type = int, help = 'whether to construct the training and test dataset')
    parser.add_argument('-M', '--MThr', dest = 'mvalue_threshold', default = 0.8, type = float, help = 'Mvalue threshold to call significant')
    parser.add_argument('-E', '--ExpThr', dest = 'gene_exp_threshold', default = 1.0, type = float, help = 'Gene expression mean threshold to binarize')
    parser.add_argument('-S', '--SdThr', dest = 'gene_sd_threshold', default = 1.0, type = float, help = 'Gene expression std threshold to binarize')
    parser.add_argument('-F', '--featureFn', dest = 'featureFn', default = 'featureList_ChIP.txt', type = str, help = 'File name where the features are stored')

    parser.add_argument('-P', '--permute', dest = 'PERMUTE', default = 0, type = int, help = 'Whether permute the dataset')
    parser.add_argument('-B', '--batch', dest = 'batch', default = 0.1, type = float, help = 'Batch size in updating')

    parser.add_argument('-I', '--maxIter', dest = 'max_iter', default = 100, type = int, help = 'Maximum iteration allowed for VI, gradient update, and batch update')
    parser.add_argument('-G', '--generalmaxIter', dest = 'general_max_iter', default = 5, type = int, help = 'Time of covering all data points')
    parser.add_argument('-e', '--eta', dest = 'eta', default = 0.1, type = float, help = 'Learning rate constant')

    parser.add_argument('-a', '--alpha0', dest = 'alpha0', default = 0.1, type = float ,help = 'L2 penalty on W')
    parser.add_argument('-A', '--alpha1', dest = 'alpha1', default = 0.1, type = float, help = 'L2 penalty on lmda')


    args = parser.parse_args()

    compute_dataset = args.compute_dataset
    mvalue_threshold = args.mvalue_threshold
    gene_exp_threshold = args.gene_exp_threshold
    gene_sd_threshold = args.gene_sd_threshold
    featurefn = args.featureFn

    PERMUTE = args.PERMUTE
    batch = args.batch

    alpha0 = args.alpha0
    alpha1 = args.alpha1

    max_iter = args.max_iter
    general_max_iter = args.general_max_iter
    general_max_iter = int( 1 / batch) * general_max_iter
    eta = args.eta

    selected_features = []
    fn = open(featurefn,'rb')
    for line in fn.readlines():
	selected_features.append(line.rstrip())
    fn.close()

    elbo_episoron = 1e-5
    obj_episoron = 1e-10
    gamma = 0.9

    main()







