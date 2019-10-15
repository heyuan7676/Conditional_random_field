import os
import sys
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

from scipy.stats import ranksums
from scipy.stats import fisher_exact

DATADIR = sys.argv[1]
PLOT = True


bins_dict = {"Quiescent":1, "ConstitutiveHet":2,"FacultativeHet":3,
             "Transcribed":4,"Promoter":5,"Enhancer":6,"RegPermissive":7,
            "Bivalent":8,"LowConfidance":9,"NoCategory":10}



def readin(chr):
    fn = os.path.join(DATADIR, 'data','%s_%s_sets.txt' % (TISSUE, chr) )
    data = pd.read_csv(fn, sep=' ')
    data['segments_segway_gene_N'] = [bins_dict[x] for x in np.array(data['segments_segway_gene_bin'])]
    data['segments_segway_snp_N'] = [bins_dict[x] for x in np.array(data['segments_segway_snp_bin'])]
    data['distance'] = data[['start','end','pos']].apply(lambda x: min(abs(x[0]-x[2]),abs(x[1]-x[2])), axis=1)
    data['functionality_score_gene'] = np.log10(np.array(data['segments_segway_gene']) + 1)
    data['functionality_score_snp'] = np.log10(np.array(data['segments_segway_snp']) + 1)
    data['hicvalues'] = np.log10(np.array(data['hicvalues']) + 1)
    data['ATAC_homer_snp'] = np.log10(data['ATAC_homer_snp'] + 1)
    data['ATAC_homer_gene'] = np.log10(data['ATAC_homer_gene'] + 1)
    data['same_compartment'] = (data['compartment_gene'] == data['compartment_snp'])
    data['same_domain'] = (data['SNP_domain'] == data['gene_domain'])
    
    return data




def test_dist(data, chr, ft, removezero=True):
    myarray_pos = np.abs(np.array(data[data['QTL']=='QTL'][ft]))
    myarray_neg = np.abs(np.array(data[data['QTL']=='neg'][ft]))
    # print "There are %f zero values in pos and %f in neg" % (sum(myarray_pos==0) *1.0/len(myarray_pos), 
    #                                                          sum(myarray_neg==0) * 1.0/len(myarray_neg))
    if removezero:
        myarray_pos = myarray_pos[myarray_pos!=0]
        myarray_neg = myarray_neg[myarray_neg!=0]
    ## Wilcoxon rank-sum test tests
    s,p = ranksums(myarray_pos, myarray_neg)
    if p < 0.05:
        print "Wilcoxon rank-sum test for the non-zero distribution for %s has pvalue %f " % (ft,p)
   
    if PLOT:
        plt.figure()
        plt.hist(myarray_pos, weights = np.ones_like(myarray_pos)/len(myarray_pos),bins=50,alpha=0.7,label="pos")
        plt.hist(myarray_neg, weights = np.ones_like(myarray_neg)/len(myarray_neg),bins=50,alpha=0.7,label="neg")
        plt.legend()
        plt.xlabel(ft)
        plt.title("Distribution of %s in positive and negative sets" % ft)
        plt.savefig(os.path.join(DATADIR,'plots','distributions_%s_%s.png' % (ft,chr)))
        plt.close()
    
    
    
    
def test_binary(data, ft):
    ### check if zero is more enriched in negative pairs
    tab = pd.crosstab(data.QTL == 'QTL', data[ft] > 0)
    if fisher_exact(tab)[1] < 0.05:
        print 'Fisher test for %s being zero in negative pairs is ' % ft, fisher_exact(tab)

    
    

def exploratory(chr):
    
    print chr
    data = readin(chr)
    
    ### check the consistency between homer results and macs results
    
    # plt.figure()
    # plt.scatter(data['ATAC_homer_gene'], data['ATAC_macs_gene'])
    # plt.plot([0,100], [0,100], ls="--", c=".3")
    # plt.show()
    # plt.close()
    
    # print 'Check the consistency between homer results and macs results:'
    # print np.corrcoef(data['ATAC_homer_snp'], data['ATAC_macs_snp'])
    # print np.corrcoef(data['ATAC_homer_gene'], data['ATAC_macs_gene'])

    # print np.corrcoef(data['CTCF_homer_snp'], data['CTCF_macs_snp'])
    # print np.corrcoef(data['CTCF_homer_gene'], data['CTCF_macs_gene'])

    ### check the distribution of features for the positive set and the negative set
    
    for ft in ['functionality_score_gene','ATAC_homer_gene','hicvalues']:
        test_dist(data, chr, ft, removezero=True)
    for ft in ['ATAC_homer_gene','same_domain']:
        test_binary(data, ft)
    
    if PLOT:
        plt.figure()
        df = data[data['QTL'] == 'QTL']
        plt.scatter(df['functionality_score_gene'],df['hicvalues'], color="blue",label = "QTL",alpha=0.5)
        df = data[data['QTL'] == 'neg']
        plt.scatter(df['functionality_score_gene'],df['hicvalues'], color="red",label = "neg", alpha=0.5)
        plt.xlabel("functionality_score_gene")
        plt.ylabel("hicvalue")
        plt.legend()
        plt.savefig(os.path.join(DATADIR,'plots','%s_2Dplots.png' % (chr)))
        plt.close()
    print




def generate_train_test(chrlist, different_chrs = False):
    if different_chrs:
        assert len(chrlist) > 1
        train_data = readin(chrlist[0])
        test_data = readin(chrlist[1])
        for c in chrlist[2:]:
            if chrlist.index(c) % 2:
                train_data = train_data.append(readin(c))
            else:
                test_data = test_data.append(readin(c))
    else:
        data = readin(chrlist[0])
        for c in chrlist[1:]:
            data = data.append(readin(c))
        ### half train, half test
        np.random.seed(0)
        train_idx = np.random.choice(len(data), len(data)/2)
        test_idx = [x for x in range(1,len(data)) if x not in train_idx]
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
    return train_data, test_data



def py_ml_dt(X,y,test_X, test_y, validation_x=None, validation_y=None, permute=False):    
    ### decision tree
    
    ### Perform cross validation 
    val,test = [],[]
    for i in range(2,20):
        clf = tree.DecisionTreeClassifier(max_depth = i, random_state=1)
        cv = ShuffleSplit(random_state = 1)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv)
        val.append(scores.mean())
        clf.fit(X, y)
        yhat = clf.predict(test_X)
        test.append(sum(np.array(yhat) == test_y) * 1.0 / len(test_y))      
    ### plot the paramter fitting process
    if not permute:
        plt.figure()
        plt.scatter(range(2,20), val,label="validation")
        plt.scatter(range(2,20), test,label="test",color="red")
        plt.legend()
        plt.xlabel("Maximum tree depth")
        plt.ylabel("Accuracy")
        plt.title("Decision tree (cross validation)")
        plt.savefig(os.path.join(DATADIR,'plots','decision_tree_cross_validation.png'))
        plt.close()
    
    ### validation using another chromosome
    if (validation_x is not None) and (validation_y is not None):
        val,test = [],[]
        for i in range(2,20):
            clf = tree.DecisionTreeClassifier(max_depth = i, random_state=0)
            clf.fit(X, y)
            yhat_vali = clf.predict(validation_x)
            val.append(sum(np.array(yhat_vali) == validation_y) * 1.0 / len(validation_y))
            yhat = clf.predict(test_X)
            test.append(sum(np.array(yhat) == test_y) * 1.0 / len(test_y))  
            
        ### don't need to plot the permuted process
        if not permute:
            plt.figure()
            plt.scatter(range(2,20), val,label="validation")
            plt.scatter(range(2,20), test,label="test",color="red")
            plt.legend()
            plt.xlabel("Maximum tree depth")
            plt.ylabel("Accuracy")
            plt.title("Decision tree (validation using other chr)")
            plt.savefig(os.path.join(DATADIR,'plots','decision_tree_validation_using_diffchr.png'))
            plt.close()
   
    optimal_i = range(2,20)[val.index(max(val))]
    clf = tree.DecisionTreeClassifier(max_depth = optimal_i, random_state=1)
    clf.fit(X,y)
    yhat = clf.predict(test_X)
    if not permute:
        print 'Optimal maximum depth is: ', optimal_i
        print "Accruancy for decision tree: ", sum(np.array(yhat) == np.array(test_y)) * 1.0 / len(test_y)
    return clf 




def py_ml_rf(X,y,test_X, test_y,validation_x=None, validation_y=None, permute=False):    
    ### random forest
    
    ### Perform cross validation
    val,test = [],[]
    for i in range(2,20):
        clf = RandomForestClassifier(max_depth = i, random_state=i)
        # Cross validation setting the seed
        cv = ShuffleSplit(random_state = i)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv)
        val.append(scores.mean())
        clf.fit(X, y)
        yhat = clf.predict(test_X)
        test.append(sum(np.array(yhat) == test_y) * 1.0 / len(test_y))   
        
    ### don't need to plot the permuted process
    if not permute:
        plt.figure()
        plt.scatter(range(2,20), val,label="validation")
        plt.scatter(range(2,20), test,label="test",color="red")
        plt.xlabel("Maximum tree depth")
        plt.ylabel("Accuracy")
        plt.title("Random forest (cross validation)")
        plt.legend()
        plt.savefig(os.path.join(DATADIR,'plots','random_forest_cross_validation.png'))
        plt.close()
    
    ### validation using another chromosome
    if (validation_x is not None) and (validation_y is not None):
        val,test = [],[]
        for i in range(2,20):
            clf = RandomForestClassifier(max_depth = i, random_state=i)
            clf.fit(X, y)
            yhat_vali = clf.predict(validation_x)
            val.append(sum(np.array(yhat_vali) == validation_y) * 1.0 / len(validation_y))
            yhat = clf.predict(test_X)
            test.append(sum(np.array(yhat) == test_y) * 1.0 / len(test_y))     
        ### don't need to plot the permuted process
        if not permute:
            plt.figure()
            plt.scatter(range(2,20), val,label="validation")
            plt.scatter(range(2,20), test,label="test",color="red")
            plt.legend()
            plt.xlabel("Maximum tree depth")
            plt.ylabel("Accuracy")
            plt.title("Random forest (validation using other chr)")
            plt.savefig(os.path.join(DATADIR,'plots','random_forest_validation_using_diffchr.png'))
            plt.close()
    
    optimal_i = range(2,20)[val.index(max(val))]
    clf = RandomForestClassifier(max_depth = optimal_i, random_state=optimal_i)
    clf.fit(X,y)
    yhat = clf.predict(test_X)
    if not permute:
        print 'Optimal maximum depth is: ', optimal_i
        print "Accruancy for random forest: ", sum(np.array(yhat) == np.array(test_y)) * 1.0 / len(test_y)
    return clf 



def py_ml_svm(X,y,test_X, test_y):    
    ### SVM
    clf = SVC()
    clf.fit(X,y)
    yhat = clf.predict(test_X)
    print "Accruancy for svm: ", sum(np.array(yhat) == np.array(test_y)) * 1.0 / len(test_y)
    return clf  




def py_ml_lg(X,y,test_X, test_y):    
    ### logistic regression
    clf = LogisticRegression(C=10000)
    clf.fit(X,y)
    yhat = clf.predict(test_X)
    print "Accruancy for logistic regression: ", sum(np.array(yhat) == np.array(test_y)) * 1.0 / len(test_y)
    return clf
    
    
    
    
    
def select_features(train_data, test_data, features, fig_fn=None, method = 'random_forest',
                    PCA_components = None, validate_using_diff_chr = False, permute=False):
    if not permute:
        print method
 
    if validate_using_diff_chr:
        vali_chr = list(train_data['chr.x'].sample(n=1, random_state=1000))[0]
        X = np.array(train_data[train_data['chr.x'] != vali_chr][features])
        y = np.array(train_data[train_data['chr.x'] != vali_chr]['QTL'])
        vali_X = np.array(train_data[train_data['chr.x'] == vali_chr][features])
        vali_y = np.array(train_data[train_data['chr.x'] == vali_chr]['QTL'])
    else:
        X = np.array(train_data[features])
        y = np.array(train_data['QTL'])
        vali_X = None
        vali_y = None
        
    test_X = np.array(test_data[features])
    test_y = np.array(test_data['QTL'])
    
    
    if PCA_components is not None:
        ### tried using PCs as features -> gave up because of lack of interpretation
        pca = PCA(n_components=PCA_components).fit(X)
        X = pca.transform(X)
        pca = PCA(n_components=PCA_components).fit(test_X)
        test_X = pca.transform(test_X)

    if method == 'random_forest':
        forest = py_ml_rf(X,y,test_X, test_y,validation_x=vali_X, validation_y=vali_y, permute=permute)
        yhat = forest.predict(test_X)
        Accruancy = sum(np.array(yhat) == np.array(test_y)) * 1.0 / len(test_y)

        # Print results for true data
        if not permute:
            importances = forest.feature_importances_
            std = np.std([tr.feature_importances_ for tr in forest.estimators_],axis=0)
            indices = np.argsort(importances)[::-1]
            # print feature importance
            print("Feature ranking:")
            for f in range(X.shape[1]):
                print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
            # Plot the feature importances of the forest
            plt.figure()
            plt.title("Feature importances in random forest")
            plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
            plt.xticks(range(X.shape[1]), features[indices],rotation=75,size=10)
            plt.xlim([-1, X.shape[1]])
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.3)
            plt.savefig(os.path.join(DATADIR,'plots','%s.png' % fig_fn))

        return Accruancy
    
    if method == 'decision_tree':
        clf = py_ml_dt(X,y,test_X, test_y,validation_x=vali_X, validation_y=vali_y, permute=permute)
        yhat = clf.predict(test_X)
        Accruancy = sum(np.array(yhat) == np.array(test_y)) * 1.0 / len(test_y)
        if not permute:
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(X.shape[1]):
                print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
            with open(os.path.join(DATADIR,'plots','%s.dot' % fig_fn), 'w') as f:
                f = tree.export_graphviz(clf, out_file=f, feature_names=features)
        return Accruancy
        
    if method == 'logistic':
        clf = py_ml_lg(X,y,test_X,test_y)
        return clf.coef_
    
    if method == 'SVM':
        clf = py_ml_svm(X,y,test_X,test_y)
    
    else:
        print "Choose the right model!"



def train_and_test(chrlist, features, different_chrs = True, useBinary = None, validate_using_diff_chr=False):
    
    train_data, test_data = generate_train_test(chrlist, different_chrs = different_chrs )
    if useBinary is not None:
        print 'binary version'
        for fb in useBinary:
            train_data[fb][train_data[fb] > 0] = 1
            test_data[fb][test_data[fb] > 0] = 1
    
    #### random forest
    train_data, test_data = generate_train_test(chrlist, different_chrs = different_chrs )
    # true data
    true_accuracy = select_features(train_data, test_data, features, fig_fn = 'rf_features_true', 
                          method = 'random_forest', validate_using_diff_chr=validate_using_diff_chr)    
    # permutation
    print 'Using permuted data (run 100 iterations):'
    accuracy = []
    for t in xrange(100):
        np.random.seed(t)
        train_data['QTL'] = np.random.permutation(train_data['QTL'])
        test_data['QTL'] = np.random.permutation(test_data['QTL'])
        accuracy.append(select_features(train_data, test_data, features, fig_fn = 'rf_features_permuted',
                                  method = 'random_forest', permute=True, validate_using_diff_chr=validate_using_diff_chr))
    print "Mean accuracy using permuted data: ", np.mean(accuracy)
    print "Empricial p-value for the true accuracy is: ", sum(true_accuracy > np.array(accuracy)) / 100.0
    print
    # draw the distribution of accuracy
    plt.figure()
    plt.hist(accuracy,bins=50, weights = np.ones_like(accuracy)/len(accuracy))
    plt.xlabel("Accuracy of permuted data")
    plt.axvline(x=true_accuracy, color="red", label="True accuracy")
    plt.legend()
    plt.savefig(os.path.join(DATADIR, 'plots','Distribution_accuracy_permuted_rf.png'))
    plt.close()
     
    #### decision tree
    train_data, test_data = generate_train_test(chrlist, different_chrs = different_chrs )
    # true data
    true_accuracy = select_features(train_data, test_data, features, fig_fn = 'dt_features_true', 
                          method = 'decision_tree', validate_using_diff_chr=validate_using_diff_chr)    
    # permutation
    print 'Using permuted data (run 100 iterations):'
    accuracy = []
    for t in xrange(100):
        np.random.seed(t)
        train_data['QTL'] = np.random.permutation(train_data['QTL'])
        test_data['QTL'] = np.random.permutation(test_data['QTL'])
        accuracy.append(select_features(train_data, test_data, features, fig_fn = 'dt_features_permuted', 
                                  method = 'decision_tree', permute=True, validate_using_diff_chr=validate_using_diff_chr))
    print "Mean accuracy using permuted data: ", np.mean(accuracy)
    print "Empricial p-value for the true accuracy is: ", sum(true_accuracy > np.array(accuracy)) / 100.0
    print
    # draw the distribution of accuracy
    plt.figure()
    plt.hist(accuracy,bins=50, weights = np.ones_like(accuracy)/len(accuracy))
    plt.xlabel("Accuracy of permuted data")
    plt.axvline(x=true_accuracy, color="red",label="True accuracy")
    plt.legend()
    plt.savefig(os.path.join(DATADIR, 'plots','Distribution_accuracy_permuted_dt.png'))
    plt.close()
 
    
    
    ### For logistic regression, use permutation to decide the significance of the coefficients
    ### this is not informative, because the logistic regression model itself is not informative
    # true coefs
    train_data, test_data = generate_train_test(chrlist, different_chrs = different_chrs )
    select_features(train_data, test_data, features, method = 'SVM') 
    # permutation
    print 'Using permuted data:'
    np.random.seed(100)
    train_data['QTL'] = np.random.permutation(train_data['QTL'])
    test_data['QTL'] = np.random.permutation(test_data['QTL'])
    select_features(train_data, test_data, features, method = 'SVM')
    print


    ### For logistic regression, use permutation to decide the significance of the coefficients
    ### this is not informative, because the logistic regression model itself is not informative
    # true coefs
    train_data, test_data = generate_train_test(chrlist, different_chrs = different_chrs )
    select_features(train_data, test_data, features, method = 'logistic') 
    # permutation
    print 'Using permuted data:'
    np.random.seed(100)
    train_data['QTL'] = np.random.permutation(train_data['QTL'])
    test_data['QTL'] = np.random.permutation(test_data['QTL'])
    select_features(train_data, test_data, features, method = 'logistic')
    print



def main():

    ### read in chrlist from txt file in the same dir
    chrlist = []
    with open('chrlist.txt', 'r') as f:
        for line in f:
            chrlist.append(line.rstrip())

    ### read in features from txt file in the same dir 
    features = []
    with open('features.txt', 'r') as ff:
        for line in ff:
            features.append(line.rstrip())
    features = np.array(['ATAC_homer_snp', 'ATAC_homer_gene',
                         'functionality_score_snp', 'functionality_score_gene',
                         'hicvalues','distance', 'same_domain'])
    
    ## explore the features
    for c in chrlist:
        exploratory(c)
    
    ## construct the model and test
    print "Cross validation"
    train_and_test(chrlist, features, different_chrs=True)
    
    print
    print "Use different validation data"
    train_and_test(chrlist, features, different_chrs=True, validate_using_diff_chr=True)
    
if __name__ == "__main__":
    TISSUE = os.environ['TISSUE']
    main()

