"""
Kernel for a Gaussian process inspired by the one described in 
1608.03343.pdf. 


The nll_fn(X, Y, noise) function is based on 
http://krasserm.github.io/2018/03/19/gaussian-processes/

https://github.com/SheffieldML/GPy

https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq


#################### Check if matrix is positive definite#######################

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

################################################################################




#######Covariant functions###############
def Distance(X, M):
	"""
	M is (D,D) symmetric matrix
	X is (D,) array
	"""
	X = X.reshape(1,-1)
	return np.matmul(np.matmul(X,M), X.reshape(-1,1))

def Matern(t,l):
    return (1+np.sqrt(5)*(t/l)+(5/3)*(t/l)**2)*np.exp(-np.sqrt(5)*(t/l))

def ExpSineSquared(t, l):
    return np.exp(-2*(np.sin(np.pi*t/l[0])/l[1])**2) #l[0] is p

def TimeCov(t, l):
	return Matern(t, l[0])*ExpSineSquared(t, l[1:])


###########################################


############Gram matrices###################
def GramMatrix(cov_func, t1,t2, l):
	"""
    t1.shape = (N,)
    t2.shape = (M,)
    Week number
    """
	N = len(t1)
	M = len(t2)
	result = np.zeros((N,M))
	for i in range(N):
		for j in range(M):
			result[i,j] = cov_func(np.abs(t1[i] - t2[j]), l)
	return result

def MaternGram(t1, t2, l):
	return GramMatrix(Matern, t1, t2, l)

def ExpSinGram(t1, t2, l):
	return GramMatrix(ExpSineSquared, t1, t2, l)    

def GramMatTime(t1, t2, l_mat, l_exp, sig2):
	return sig2[0]* MaternGram(t1, t2, l_mat) + sig2[1]* ExpSinGram(t1, t2, l_exp)
    
def RBF(X1, X2, M):
	return GramMatrix(lambda X, l:  np.exp(-Distance(X,M)), X1, X2, M)

def TimeCovGram(t1, t2, l):
	return GramMatrix(TimeCov, t1, t2, l)

def InnProdKernel(X1, X2, l2_met):
    """
    X1.shape = (N,D)
    X2.shape = (M,D)
    l2 is (D,) array  of positive scalars. 
    """
    L = np.diagflat(l2_met)
    return np.matmul(np.matmul(X1,L), X2.T)
#############################################################



##########################Kernels#############################
def Kernel(X1, X2, l2_time, sig2, l2_met):
	"""
	l2 is (D,) array
	Can also introduce a (D,1) array Lambda and take 
	M = Lambda.T Lambda + diag(l2_met)
	"""
	M = np.diagflat(l2_met)
	return sig2[0]*MaternGram(X1[:,0], X2[:,0], l2_time[0]) + \
	sig2[1]*TimeCovGram(X1[:,0], X2[:,0], l2_time[1:]) + \
	sig2[2]*RBF(X1[:,1:], X2[:,1:], M)


## This is the Kernel from the [1608.03343.pdf] paper
#def Kernel(X1, X2, l2_time, sig2, l2_met):
#	return sig2[0]*MaternGram(X1[:,0], X2[:,0], l2_time[0]) + \
#	sig2[1]*TimeCovGram(X1[:,0], X2[:,0], l2_time[1:]) + \
#	sig2[2]*InnProdKernel(X1[:,1:], X2[:,1:], l2_met)
###################################################





#############Negative log-likelihood##################
def nll_fn(X, Y, noise2):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X and Y and given noise^2 level.
    Args:
        X.shape = (M, D+1).
        Y.shape = (M,1).   
        noise2: known noise^2 level of Y.  
        naive: if True use a naive implementation of objective function, 
        	   if False use a numerically more stable implementation.   
    Returns:
        Minimization objective.
    '''
    M = X.shape[0]
    D = X.shape[1]-1
    def nll_stable(theta):
    	l2_time = theta[:4]
    	sig2 = theta[4:7]
    	#Lambda = theta[7:7+D]	
    	l2_met = theta[7:]
    	A = Kernel(X, X, l2_time, sig2, l2_met)
    	#np.linalg.eigvalsh(A)[0] = smallest eigenvalue of A
    	error = -np.minimum(np.linalg.eigvalsh(A)[0], 0) + 1e-03 
    	K = A + (error + noise2) * np.eye(M)
    	L = cholesky(K)
    	return np.sum(np.log(np.diagonal(L))) + 0.5 * Y.T.dot(lstsq(L.T, lstsq(L, Y)[0])[0])
    
    return nll_stable
#######################################################


###################Posterior predictive#################
def posterior_predictive(Xs, X, Y, l2_time, sig2, l2_met, noise2=0.2):
    ''' 
    Mean and covariance matrix of  posterior predictive distribution 
    from m training data X and Y and n new inputs Xs.
        Xs.shape = (N,D+1).
        X.shape = (M,D+1).
        Y.shape = (M,1) (NOT (M,))!!
        noise: Noise parameter.
    Returns:
        Posterior mean vector (N,D) and covariance matrix (N,N).
    '''

    K = Kernel(X, X, l2_time, sig2, l2_met) + noise2 * np.eye(X.shape[0])
    Ks = Kernel(Xs, X, l2_time, sig2, l2_met)
    Kss = Kernel(Xs, Xs, l2_time, sig2, l2_met)
    K_inv = inv(K)
    mu_s = np.matmul(np.matmul(Ks,K_inv), Y)
    cov_s = Kss - np.matmul(np.matmul(Ks, K_inv), Ks.T)
    
    return mu_s, cov_s
##############################################################


################Data preprocessing##########################
def fitting_preprocess(Xy_train, features):
    """
    Xy_train is training data and labels.
    Features is a list of columns of Xy_train
    Returns:
         X: (see below)
         y: normalized labels, with zero mean. 
         mu: mean of labels
    """
    columns = [('week_number', '')] + features  
    X = Xy_train[columns].values
    log_Yplus1 = np.log(1+Xy_train[('labels', '')].values)
    gamma = np.mean(log_Yplus1, axis = 0)
    Y = log_Yplus1 - gamma
    
    return X, Y, gamma
###############################################################



################################################################
def Y_pred(Xs, X, Y, gamma, theta, std_dev = False):
    """
    Returns the predicted number of cases per week, given the parameter theta. 
    """
    l2_time = theta[:4]
    sig2 = theta[4:7]
    l2_met = theta[7:]
    mu_s, cov_s = posterior_predictive(Xs, X, Y, l2_time, sig2, l2_met)
    
    least_eigenval = np.linalg.eigvalsh(cov_s)[0]
    if least_eigenval < 0:
        print('Error: Covariant matrix not positive definite')
        sys.exit(1)
    
    Y_pred = np.exp(mu_s + gamma)-1
    
    """
    95% confidence interval
    """
    if std_dev:
        uncertainty = 1.96 * np.sqrt(np.diag(cov_s))
        Y_plus = np.exp(mu_s + uncertainty + gamma)-1
        Y_minus = np.exp(mu_s - uncertainty + gamma)-1
        return Y_minus, Y_pred, Y_plus
        
    return Y_pred 
####################################################################


###########################Score####################################
def score(Y_test, Ys):
    """
    Y_test and Y_pred are np.array of shape (-1, 1)
    """
    u = ((Y_test - Ys) ** 2).sum(axis = 0)
    v = ((Y_test - Y_test.mean()) ** 2).sum(axis = 0)
    score = 1-u/v
    return score[0]
####################################################################




