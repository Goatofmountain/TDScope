''''
__Author__: Kailing Tu
__Version__: v4.0.0
__ReleaseTime__: 2024-03-25
Requirement:
    numpy v1.21.5
    matplotlib v3.4.3
    scipy v1.7.3
Description:
Modular components:
    CallDistance:   
        Function calculate the Jaccard distance between two aligned sequence
    pariwiseDistance: 
        Function calulate the Jaccard distance matrix 
    CheckTheta:
        Function for theta parameter checking, add a small value(1e-10 by default) on theta paramter matrix to avoid 0 or 1 prob value 
    par_init:
        Function for clustering initialization
    loglik:
        Function for log likelihood calculation 
    gamma_updating:
        Function for gamma parameter updating (the E step)
    pitheta_updating:
        Function for pie and theta parameter updating (the M step)
    EM:
        Function for EM iteration change to iterate 40 times for time cliping 
    BIC:
        Function for Bayesian Information Criterion score calculation: 2 * log(Lik) - n_theta * log(N)
    EMCluster:
        Main Function for EM clustering of different K value, by default max K would be set to 9
    TKLCluster:
        Same as EMCluster but more likelihood and parameter visualization during EM step, Function for the author (TKL) to debug
'''


import numpy as np
## EM Clustering 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt 
import time 

np.random.seed(2023)

def CallDistance(read1, read2):
    # calculate reads jaccard distance 
    total = len(read1)
    common = np.where(read1==read2)[0].shape[0]
    return(common / total)

def pariwiseDistance(seqdatamx):
    # calculate distance matrix for clustering 
    N = seqdatamx.shape[0]
    DistanceClust = np.eye(N)
    for i in range(seqdatamx.shape[0]):
        for j in range(i):
            DistanceClust[i,j]=DistanceClust[j,i]=CallDistance(seqdatamx[i], seqdatamx[j])
    return(DistanceClust)

def CheckTheta(thetap_t, avoid=1e-10):
    '''
    Check theta value to avoid zero or negative value 
    '''
    adjtheta = thetap_t.copy()
    adjtheta[np.where(adjtheta==0)] += avoid
    adjtheta[np.where(adjtheta==1)] += -1*avoid
    return(adjtheta)

def CheckParam(AimParam, episilon=1e-10):
    '''
    to avoid 0 or 1 parameter 
    '''
    return(np.clip(AimParam, episilon, 1-episilon))

def par_init(K,nf,initselection,seqdatamx=None,ClusterF=None, Z=None,alphaLen=5):
    '''
    init parameters
    input:
        K:  the number of clusters
        nf: the number of features
        initselection:  Type of initilization methods, default set to 0 represents the true value initilization, set to 1 represents to prob value after simple reads clustering, set to other represents random value initilization,
        ClusterF:   K*nf dim ClusterFeature matrix recording the successful probability of each feature in group k.
        Z: hierarchy cluster tree result based on reads Jaccard distance matrix made by pariwiseDistance function in TKLScanner
    return:
        pi_0:   K length vector, representing for the weights for each cluster
        thetap_0: K*nf*5 matrix, representing for the success probability of each Features in different cluster k
    '''
    pi_0 = np.repeat(1/K, K)
    if initselection == 0:
        thetap_0 = ClusterF
    elif initselection == 1:     # Take hierarchy clustering result for model init 
        N = seqdatamx.shape[0]
        readsClust = fcluster(Z, K, criterion='maxclust')
        gamma = np.zeros((N,K))
        for idx in range(len(readsClust)):
            gamma[idx,readsClust[idx]-1] = 1
        pi_0,thetap_0 = pitheta_updating(K,gamma, seqdatamx)
    else:
        thetap_0 = np.stack([np.random.dirichlet(np.ones(alphaLen), size=nf) for k in range(K)])
    return(pi_0, thetap_0)

# Likelihood calculation for each read 
def loglik(pic,theta,gamma,seqdatamx):
    '''
    Calculate the likelihood function L(pie,theta|data) based on binormal data distribution
    input:
        pic:    K length vector, representing for the weights for each cluster
        theta:  K*nf*alpha matrix, representing for the success probability of each Features in different cluster k
        seqdatamx:  N*nf matrix, recording the true seq data information with 0 or 1 valued for N reads nf feature.
    output:
        likMean:    Mean likelihood value for N reads.
    Likelihood for each read: log(exp(log(pie_k) + log(L(x_i|theta_k))) + ... + exp(log(pie_0) + log(L(x_i|theta_0))))
    '''
    theta1 = CheckParam(theta, 1e-10)
    N, nf = seqdatamx.shape
    K = pic.shape[0]
    seqdatamx_onehot = np.array([np.eye(theta1.shape[-1])[seqdatamx[I]] for I in range(seqdatamx.shape[0])])
    perReadLik = np.zeros((N,))
    for k in range(K):
        perReadLik += ((np.log(theta1[k]) * seqdatamx_onehot).sum(axis=2).sum(axis=1) + np.log(CheckParam(pic[k]))) * gamma[:,k]
    return(perReadLik)

## Gamma Updating (E-step)
def safe_exp(x, min_val=-700, max_val=700):
    """
    Safe exp calculation 
    """
    x_clipped = np.clip(x, min_val, max_val)
    return np.exp(x_clipped)

def gamma_updating(K, pi_t, thetap_t, seqdatamx):
    '''
    Update gamma according to pi and theta parameter in the last step
    Input:
        pi_t:K length vector, representing for the weights for each cluster
        thetap_t:K*nf matrix, representing for the success probability of each Features in different cluster k
        seqdatamx:  N*nf*alpha matrix, recording the true seq data information with 0 or 1 valued for N reads nf feature.
        ReadFeatureExist: N*nf matrix, recording whether specific feature exists (1) in reads or not (0)
    output:
        gamma_t: updated gamma matrix (N*K) represents for the likelihood of reads assigned to the K's group, 
                 which could also be considered as the expectation of clustering result according to pi,theta
    '''
    N, nf = seqdatamx.shape
    gamma_t_updated = np.zeros((N,K))
    thetap_t_adj = CheckParam(thetap_t, 1e-10)
    MarginProb_log = np.zeros((N,K))
    for alpha in range(thetap_t_adj.shape[-1]):
        MarginProb_log += np.dot(np.where(seqdatamx==alpha, 1, 0), np.log(thetap_t_adj[:,:,alpha].T))
    MarginProb_log += np.log(pi_t.reshape((pi_t.shape[0],1)).T)
    gamma_t_list = []
    for I in range(MarginProb_log.shape[1]):
        gamma_t_list.append(1 / safe_exp(MarginProb_log - MarginProb_log[:,I].reshape((MarginProb_log.shape[0],1))).sum(axis=1))
    gamma_t_updated = np.vstack(gamma_t_list).T  
    return(gamma_t_updated)
    # MarginProb = np.exp(MarginProb_log)
    # JointProb = MarginProb.sum(axis=1).reshape((MarginProb.shape[0], 1))
    # gamma_t_updated = MarginProb/JointProb 
    # return(gamma_t_updated)

## Pi and Theta updating (M-step)
def pitheta_updating(K, gamma_t, seqdatamx,alphaLen=5):
    '''
    Update pi and theta parameter according to gamma in the last step
    Input:
        gamma_t: updated gamma matrix (N*K) represents for the likelihood of reads assigned to the K's group, 
                 which could also be considered as the expectation of clustering result according to pi,theta
        seqdatamx:  N*nf matrix, recording the true seq data information with 0 or 1 valued for N reads nf feature.
        ReadFeatureExist: N*nf matrix, recording whether specific feature exists (1) in reads or not (0)
    Output:
        pi_t:K length vector, representing for the weights for each cluster
        thetap_t:K*nf*alpha matrix, representing for the success probability of each Features in different cluster k
    '''
    N,nf = seqdatamx.shape
    ReadFeatureExist = np.where(np.isnan(seqdatamx), 0, 1)
    pi_t = gamma_t.sum(axis=0)/N
    thetap_t = np.zeros((K, nf, alphaLen))
    # possible bug for pi
    bug_pi1 = np.where(pi_t*N<1)[0]
    bug_pi2 = np.where(np.isnan(pi_t))[0]
    # print("pi: %s\tbugpi: %s" % (pi_t,bug_pi1))
    gammaSum = np.dot(gamma_t.T, ReadFeatureExist)
    if (len(bug_pi1)==0) and (len(bug_pi2)==0):
        thetap_t = np.dstack([np.dot(gamma_t.T, np.where(seqdatamx==alpha, 1, 0))/gammaSum for alpha in range(alphaLen)])
    else:
        pi_t = np.repeat(1/K, K)
        thetap_t = np.stack([np.random.dirichlet(np.ones(5), size=nf) for k in range(K)])
    return(pi_t, thetap_t)

def EM(K, seqdatamx, initselection=1, Nstep=20,Z=None):
    nf = seqdatamx.shape[1]
    Nstep = Nstep
    pi_0, thetap_0 = par_init(K,nf,initselection=initselection,seqdatamx=seqdatamx, Z=Z, ClusterF=None)
    gamma_0 = gamma_updating(K,pi_0, thetap_0, seqdatamx)
    gamma_t = gamma_0.copy()
    ParamDict = {"pi":[pi_0], "theta":[thetap_0], "gamma":[gamma_0], "likelihood":[]}
    # iteration 
    for n in range(Nstep):
        start_time = time.time()
        pi_t, thetap_t = pitheta_updating(K, gamma_t, seqdatamx)
        gamma_t = gamma_updating(K,pi_t, thetap_t, seqdatamx)
        ParamDict['pi'].append(pi_t)
        ParamDict['theta'].append(thetap_t)
        ParamDict['gamma'].append(gamma_t)
        TMP_Likelihood = loglik(pi_t,thetap_t,gamma_t,seqdatamx)
        ParamDict['likelihood'].append(TMP_Likelihood)
        end_time = time.time()
        # print('%s: EM iteration for %s Step %s finished with %s min ' % (time.ctime(), K, Nstep+1, (end_time-start_time)/60))
    return(ParamDict)

def BIC(ParamDict, ZeroParamNum=0):
    # Calculate BIC 
    theta_f = ParamDict['theta'][-1]
    pi_f = ParamDict['pi'][-1]
    n_theta = len(pi_f) - 1 + theta_f.shape[0] * theta_f.shape[1] * (theta_f.shape[2]-1) - ZeroParamNum
    Rlik = ParamDict['likelihood'][-1]
    N = len(Rlik)
    Result = 2 * Rlik.sum() - n_theta * np.log(N)
    return(Result)

def EMCluster(seqdatamx,initselection=1,max_C=9, ShowPlot=False):
    '''
    Complete pipeline for reads clustering, by default the max cluster Number would be 9, each K should cluster for 
    '''
    ### Get seqdata information for accuracy BIC Score 
    FeatureCount = []
    for Label in range(5):
        tmpFeatureCount = np.zeros((seqdatamx.shape[1],))
        ColIDX,Count = np.unique(np.where(seqdatamx==Label)[1], return_counts=True)
        for I in range(ColIDX.shape[0]):
            tmpFeatureCount[ColIDX[I]] = Count[I]
        FeatureCount.append(tmpFeatureCount)
    FeatureCountArr = np.sort(np.array(FeatureCount), axis=0)
    ZeroParamNum = np.where(FeatureCountArr==0)[0].shape[0]
    ### Get seqdata information for accuracy BIC Score 
    BICList = []
    ParameterPool = []
    Kmin, Kmax = 1,np.min([max_C+1, seqdatamx.shape[0]])
    # calculate pairwise distance matrix if initselection==1
    Z = None
    if initselection==1:   # initialization cluster with hierarchy result 
        seqdata_dist = pariwiseDistance(seqdatamx)
        Z = linkage(seqdata_dist,'ward')
    if ShowPlot:
        plt.figure(figsize=(9,9))
    for K in range(Kmin,Kmax):
        BICNum = np.nan
        ClusterCount = 5
        while np.isnan(BICNum) and (ClusterCount!=0):
            ParamDict = EM(K,seqdatamx, initselection=initselection, Z=Z)
            BICNum = BIC(ParamDict)
            ClusterCount += -1
        # Test process 
        if ShowPlot:
            plt.subplot(3,3,K)
            plt.plot(np.arange(50), [np.mean(x) for x in ParamDict['likelihood']])
            plt.title('K = %s' % K)
            if K in [1,4,7]:
                plt.ylabel('mean log likelihood')
            plt.show()
        BICList.append(BIC(ParamDict, ZeroParamNum))
        gamma_t = ParamDict['gamma'][-1]
        ParameterPool.append(ParamDict)
    MAXBIC = np.nanargmax(np.array(BICList))
    ResultClust = MAXBIC
    # ResultClust = 0
    K = ResultClust+1
    # Avoid Close BIC between K=1 and K=2 
    if K == 1:
        if BICList[0] - BICList[1] <= seqdatamx.shape[1] * np.log(seqdatamx.shape[0]):
            K = 2
            ResultClust = 1
    gamma = ParameterPool[ResultClust]['gamma'][-1]
    thetap = ParameterPool[ResultClust]['theta'][-1]
    pie = ParameterPool[ResultClust]['pi'][-1]
    Rclust = np.argmax(gamma, axis=1)
    return([K, seqdatamx, Rclust, thetap, gamma, pie, np.array(BICList)])

def TKLCluster(seqdatamx,initselection=0,Z=None):
    '''
    Test Cluster function, same setting as EMCluster function, but more parameter plot for test 
    '''
    ### Get seqdata information for accuracy BIC Score 
    FeatureCount = []
    for Label in range(5):
        tmpFeatureCount = np.zeros((seqdatamx.shape[1],))
        ColIDX,Count = np.unique(np.where(seqdatamx==Label)[1], return_counts=True)
        for I in range(ColIDX.shape[0]):
            tmpFeatureCount[ColIDX[I]] = Count[I]
        FeatureCount.append(tmpFeatureCount)
    FeatureCountArr = np.sort(np.array(FeatureCount), axis=0)
    ZeroParamNum = np.where(FeatureCountArr==0)[0].shape[0]
    ### Get seqdata information for accuracy BIC Score 
    BICList = []
    ParameterPool = []
    Kmin, Kmax = 1,np.min([10, seqdatamx.shape[0]])
    # calculate pairwise distance matrix 
    seqdata_dist = pariwiseDistance(seqdatamx)
    Z = linkage(seqdata_dist,'ward')
    # Make hi
    plt.figure(figsize=(9,9))
    for K in range(Kmin,Kmax):
        BICNum = np.nan
        ClusterCount = 5
        while np.isnan(BICNum) and (ClusterCount!=0):
            ParamDict = EM(K,seqdatamx, initselection=initselection, Z=Z)
            BICNum = BIC(ParamDict)
            ClusterCount += -1
        # Test process 
        plt.subplot(3,3,K)
        plt.plot(np.arange(40), [np.mean(x) for x in ParamDict['likelihood']])
        plt.title('K = %s' % K)
        if K in [1,4,7]:
            plt.ylabel('mean log likelihood')
        BICList.append(BIC(ParamDict, ZeroParamNum))
        gamma_t = ParamDict['gamma'][-1]
        ParameterPool.append(ParamDict)
    plt.show()
    MAXBIC = np.nanargmax(np.array(BICList))
    ResultClust = MAXBIC
    # ResultClust = 0
    K = ResultClust+1
    # Avoid Close BIC between K=1 and K=2 
    if K == 1:
        if BICList[0] - BICList[1] <= seqdatamx.shape[1] * np.log(seqdatamx.shape[0]):
            K = 2
            ResultClust = 1
    gamma = ParameterPool[ResultClust]['gamma'][-1]
    thetap = ParameterPool[ResultClust]['theta'][-1]
    pie = ParameterPool[ResultClust]['pi'][-1]
    Rclust = np.argmax(gamma, axis=1)
    return([K, seqdatamx, Rclust, thetap, gamma, pie, np.array(BICList)])



