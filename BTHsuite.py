'''
Goal: given genotypes x and phenotypes y, determine if the samples y
are heteroskedastic with behavior detemined by the covariates x
'''


####################################################################################################
##########################			    package requierements   		  ##########################
####################################################################################################
# Python version: 
from scipy import stats
from scipy import optimize
import numpy as np
import math
import random
import sys
import os
import warnings
from scipy.stats import loglaplace
from scipy.stats import cauchy
##########################                   dependencies                 ##########################
from simulations import *
##########################                global hyperparams              ##########################
global M, b, theta1, theta2, v, lam, logexp_param, genotype, phenotype, gamma


########################################################################################
##################################     Ex Run       ####################################
########################################################################################

def test_BTH():
    '''
    creates a test sample, and then computes the corresponding bayes factors for the permuted 
    and unpermuted data sets
    '''
    [maf, n, intercept, beta, sigma2, logalpha,T,c]  = [0.2, 300,0, 0.1, 1., 0.2, 1, 0 ]
    [genotype, phenotype] = sim_xy(maf, n, intercept, beta, sigma2, logalpha,T,c)
    logBF = BTH(genotype,phenotype)
    [genotypePermuted, phenotypePermuted] = special_shuffle2(genotype,phenotype)
    logBFperm = BTH(genotypePermuted, phenotypePermuted)

    return [logBF, logBFperm]


########################################################################################
##################################     Levene       ####################################
########################################################################################

def Lev(x,y):
    ''' levene works only for 0, 1 or 2
    '''

    roundedX = [int(round(x[i])) for i in range(len(x))]
    sample0 = [y[i] for i in range(len(x)) if int(roundedX[i])==0]  # y0
    sample1 = [y[i] for i in range(len(x)) if int(roundedX[i])==1]  # y1        
    sample2 = [y[i] for i in range(len(x)) if int(roundedX[i])!=0 and int(roundedX[i])!=1] #y2

    if len(sample0)==0:
        return -5
    if len(sample1)==0:
        return -5
    if len(sample2)==0:
        return -5
    
    N = len(x)
    k = 3
    mean_sample0 = np.mean(sample0) # mean for levean, median for BF
    mean_sample1 = np.mean(sample1)
    mean_sample2 = np.mean(sample2)

    z0 = [ np.abs( sample0[j] - mean_sample0) for j in range(len(sample0))]
    z1 = [ np.abs( sample1[j] - mean_sample1) for j in range(len(sample1))]
    z2 = [ np.abs( sample2[j] - mean_sample2) for j in range(len(sample2))]

    zdd =1./N *( np.sum(z0) + np.sum(z1) + np.sum(z2))
    z0d = np.mean(z0)
    z1d = np.mean(z1)
    z2d = np.mean(z2)

    W1 = (N - k)* 1./ (k - 1) * ( len(z0) *(z0d - zdd)**2 + len(z1) * ((z1d - zdd)**2) + len(z2) * ((z2d - zdd)**2))
    W2 = 1.* ( np.sum([(z0[j] - z0d)**2 for j in range(len(z0))]) + np.sum([(z1[j] - z1d)**2 for j in range(len(z1))]) + np.sum([(z2[j] - z2d)**2 for j in range(len(z2))]))
    return  W1/W2

########################################################################################
##################################  Brown Forsythe  ####################################
########################################################################################

def BrFor(x, y):
    '''
    TO DO from run_brown_forsythe write code that can readily be changed to Julia
    '''

    roundedX = [int(round(x[i])) for i in range(len(x))]
    sample0 = [y[i] for i in range(len(x)) if int(roundedX[i])==0]  # y0
    sample1 = [y[i] for i in range(len(x)) if int(roundedX[i])==1]  # y1
    sample2 = [y[i] for i in range(len(x)) if int(roundedX[i])!=0 and int(roundedX[i])!=1] #y2
    
    if len(sample0)==0:
        return -5.
    if len(sample1)==0:
        return -5.
    if len(sample2)==0:
        return -5.

    N = len(x)
    k = 3
    
    z0 = [ np.abs( sample0[j] - np.median(sample0)) for j in range(len(sample0))]
    z1 = [ np.abs( sample1[j] - np.median(sample1)) for j in range(len(sample1))]
    z2 = [ np.abs( sample2[j] - np.median(sample2)) for j in range(len(sample2))]
    
    zdd =1./N *( np.sum(z0) + np.sum(z1) + np.sum(z2)) 
    z0d = np.mean(z0)
    z1d = np.mean(z1)
    z2d = np.mean(z2)
    
    W1 = (N - k)* 1./ (k - 1) * ( len(z0) *(z0d - zdd)**2 + len(z1) * ((z1d - zdd)**2) + len(z2) * ((z2d - zdd)**2)) 
    W2 = 1.* ( np.sum([(z0[j] - z0d)**2 for j in range(len(z0))]) + np.sum([(z1[j] - z1d)**2 for j in range(len(z1))]) + np.sum([(z2[j] - z2d)**2 for j in range(len(z2))])) 
        
    return W1/W2

########################################################################################
##################################  HLS eLife  #########################################
########################################################################################
def TSLS(snp,out):
    # implements the test found in eLife publications, Brown et al, Durbin last
    x = np.array(snp)
    y = np.array(out)
    A = np.array([ x, np.ones(len(x))])
    w = np.linalg.lstsq(A.T,y)[0]
    residues = y - w[0]*x - w[1]
    residues = residues**2
    [test_stat, pvalue] = stats.spearmanr(residues,x)
    return abs(test_stat)


def special_shuffle2(genotype,phenotype):
    x = np.array(genotype)
    y = np.array(phenotype)
    A = np.array([x, np.ones(len(x))])
    w = np.linalg.lstsq(A.T,y)[0]
    residues = y - w[0]*x -w[1]
    random.shuffle(residues)
    random.shuffle(residues)

    residues = np.array(residues)    
    
    ww = np.linalg.lstsq(A.T,residues)[0]
    residues_final = residues - ww[0]*x -ww[1]
    yhat = residues_final + w[0]*x + w[1]
    return [genotype,yhat]

####################################################################################################
############################################  main  ################################################
####################################################################################################
def BTH(genotype, phenotype):
    '''
    returns a logarithmic Bayes Factor assesing the heteroskedasticity of the genotype vector 
    with respect to the regressor given by the phenotype vector

         P (phenotype | genotype,  log alpha != 0)
    BF = -----------------------------------------
    	 P (phenotype | genotype,  log alpha = 0 )

    @input
    genotype - array of length n
    phenotype - array of length n
    @phenotypeput
    logBF - a real number

    Interpretability:
    depends on: INLACauchy_log_Laplace
    '''
    set_global_params()
    warnings.simplefilter("ignore")
    # compute the numerator of the BF: P (phenotype | genotype,  log alpha != 0)
    [numerator, alpha1, inter1, beta1,sigma1] = INLACauchy_log_Laplace(genotype,phenotype,theta1, theta2, v, b, 1,gamma)
    # compute the denominator of the BF: P (phenotype | genotype,  log alpha = 0)
    [denominator, alpha2, inter2, beta2, sigma2] = INLACauchy_log_Laplace(genotype,phenotype,theta1, theta2, v, b, 0,gamma) 

    # compute base 10 log Bayes Factor
    logBF  = (numerator - denominator) * np.log10(np.e)
    return logBF

####################################################################################################
####################################     helper functions     ######################################
####################################################################################################

def INLACauchy_log_Laplace(genotype, phenotype,theta1, theta2, v,lam, int_type,gamma):
    '''
    if int_type = 1, it evaluates the numerator, integrating over the heteroskedastic parameter alpha
    if int_type = 0, it evaluates the denominator, setting log alpha to 0

    @phenotypeput
    log_laplace_term (numerator/denominator), 
    MAP estimates: alpha_hat, beta0_hat, beta_hat,sigma_hat
    --------------------------------------------------------------------------------------------------
    dependends on: INLACauchy_h (evaluation offirst order derivatives), 
                   INLACauchy_h_hess(evaluation of the hessian), 
                   INLACauchy_hprime(first order derivative),
                   INLACauchy_hhprime(second order derivative)
    '''

    N = len(genotype)

    # set bounds for the integral estimation
    if int_type == 1:
        bound1 = 0.00000000000001     # for Cauchy, add small step for numerical errors 
        bound2 = None     # for Cauchy
    elif int_type == -1:
        bound1 = 1.     # does not matter for Cauchy
        bound2 = None   # does not matter for Cauchy
    else:
        bound1 = None
        bound2 = None

# MAP estimates 
    N = len(phenotype)
    params = [phenotype, genotype, theta1, theta2, v, lam, N, int_type,gamma]
    ##############################################################
    if int_type != 0:
        # perform triple integral estimation with prior over alpha
        if int_type == -1:
            ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1.,1.,0.], fprime= lambda x: INLACauchy_hprime(x,params), \
                                bounds=((bound1, bound2),(0.00001,None),(None,None)),\
                                epsilon =1e-5, disp = False)
        else:
            ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1.,1.,0.], fprime= lambda x: INLACauchy_hprime(x,params), \
                                bounds=((bound1, bound2),(0.000000000001,None),(None,None)),\
                                epsilon =1e-5,disp = False)
        
        
        [alpha_hat, sigma_hat, beta0_hat] = ans[0]
        
        evaluate_h = INLACauchy_h([alpha_hat, sigma_hat, beta0_hat], params) 
        evaluate_hess = INLACauchy_h_hess([alpha_hat, sigma_hat, beta0_hat], params) # fing the values of the hessian terms at MAP  
        d = 3.
        
        S2 = sum([(genotype[i]**2) * (alpha_hat ** genotype[i]) for i in range(len(genotype))])
        S1 = sum([genotype[i] * (alpha_hat ** genotype[i]) for i in range(len(genotype))])
        Q1 = sum([genotype[i] * phenotype[i] * (alpha_hat ** genotype[i]) for i in range(len(genotype))]) 
        
        
        beta_hat = 1./(v + 1./sigma_hat * S2) * 1./sigma_hat *(Q1 - beta0_hat*S1)
    else:
        ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1., 0.00000001], fprime= lambda x: INLACauchy_hprime(x,params), \
                                bounds=((0.00001, None),(None, None)),epsilon =1e-5,disp =False)
        [sigma_hat,beta0_hat,] = ans[0]

        evaluate_h = INLACauchy_h([sigma_hat, beta0_hat], params) # find the value of the h function at the MAP estimates
        evaluate_hess = INLACauchy_h_hess([sigma_hat,beta0_hat],params) # fing the values of the hessian terms at MAP   
        d = 2.
        alpha_hat = 1.
        S2 = sum([(genotype[i]**2) * (alpha_hat ** genotype[i]) for i in range(len(genotype))])
        S1 = sum([genotype[i] * (alpha_hat ** genotype[i]) for i in range(len(genotype))])
        Q1 = sum([genotype[i] * phenotype[i] * (alpha_hat ** genotype[i]) for i in range(len(genotype))]) 
               
        beta_hat = 1./(v + 1./sigma_hat * S2) * 1./sigma_hat *(Q1 - beta0_hat*S1)
     
    log_laplace_term = (- N * evaluate_h) + d/2. * np.log(2*np.pi) - \
     0.5 * np.log(abs(evaluate_hess)) - d/2. *np.log(N) 
    return [log_laplace_term, alpha_hat, beta0_hat, beta_hat,sigma_hat]

####################################################################################################
def INLACauchy_h(x,params):
    '''
    x = [alphaHat, sigmaHat, beta0Hat]

    Evaluates the main function h, which enters the Laplace approx through the term exp(- N * h)
    depends on: INLACauchy_intermediary_params
    '''

    [genotype, phenotype, theta1, theta2, v, lam, N,int_type,gamma] = params
        
    if int_type != 0:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        
        if int_type == -1: #will not matter for Cauchy
            prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (1./lam - 1.) * np.log(alpha)
        else:
            prior_on_alpha = - np.log(np.pi * gamma + np.pi / gamma * ((np.log(alpha))**2)) - np.log(alpha) # Cauchy
        
    else:
        alpha = 1.
        sigma = x[0] 
        beta0 = x[1]
        prior_on_alpha = 0.
      
    [S, R, Q, O] = INLACauchy_intermediary_params(x, genotype, phenotype,int_type)
    
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    G = sigma * v + S2
    prior_on_sigma =  (-theta1 -1)*np.log(sigma) - theta2*1./sigma

    if G ==0.:
        #print('error at params, G, h')
        #rint(params)
        G = 0.00000001
        
    if sigma == 0.:
        #print('error at params, sigma, h')
        #print(params)
        sigma = 0.00000001
    
    L0 = (-N/2. + 0.5) * np.log(sigma) - 0.5 * np.log(G) + 0.5 * np.sum(genotype) * np.log(alpha)
    L1 =  -1./(2 * sigma) *(R2 - Q1*Q1/G)
    L2 = 1./sigma * beta0 *(R1 - Q1*S1/G)
    L3 = - 1./(2 * sigma) *beta0 * beta0 * (S0 - S1*S1/G)
     
    h = -1./N *(L0 + L1 + L2 + L3 + prior_on_alpha + prior_on_sigma)
    
    return h

####################################################################################################
def INLACauchy_hprime(x,params):
    ''' 
    compute the first order derivative
    depends on: INLACauchy_intermediary_params
    
    '''
    [genotype, phenotype, theta1, theta2, v, lam, N, int_type,gamma] = params
    if int_type != 0:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        if int_type == -1:
            d_alpha_prior_on_alpha = (1./lam -1.)*1./alpha # not used for Cauchy
        else:
            d_alpha_prior_on_alpha =  - (gamma**2 + (np.log(alpha))**2 + 2 * np.log(alpha))/( gamma**2 *alpha + alpha * (np.log(alpha))**2)#for Cauchy
    else:    
        alpha = 1.
        sigma = x[0]
        beta0 = x[1]

        
    [S, R, Q, O] = INLACauchy_intermediary_params(x, genotype, phenotype, int_type)
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    
    G = sigma * v + S2
    
    if G ==0.:
        #print('error at params, G, hprime')
        #print(params)
        G = 0.00000001
        
    if sigma ==0.:
        #print('error at params, sigma, hprime')
        #print(params)
        sigma = 0.00000001
        
    if alpha ==0:
        #print('error at alpha,hprime')
        #print(params)
        alpha = 0.00000001
    

    ### d_sig_L0
    d_sig_L0 = (-0.5*N + 0.5) * 1./sigma - 0.5 * v /G     
    ### d_sig_L1
    d_sig_L1 = 1./(2* sigma*sigma)*(R2 - Q1*Q1/G) - 1./(2*sigma) * v * Q1*Q1/(G*G)   
    ### d_sig_L2
    d_sig_L2 = -1./(sigma*sigma) *beta0 *( R1 - Q1*S1/G) + 1./sigma * beta0 * Q1* S1*v/ (G*G)      
    ### d_sig_L3
    d_sig_L3 = 1./ (2* sigma*sigma) * beta0* beta0 *(S0 - S1*S1/G) - 1./(2*sigma) *beta0*beta0 *S1*S1*v/(G*G)  
    ### d_sig_prior_on_alpha
    d_sig_prior_on_alpha = 0 
    ### d_sig_prior_on_sigma
    d_sig_prior_on_sigma = (-theta1 -1) *(1./sigma) + theta2/(sigma*sigma)   
    
    ### d_beta0_L0
    d_beta0_L0 = 0   
    ### d_beta0_L1
    d_beta0_L1 = 0   
    ### d_beta0_L2
    d_beta0_L2 = 1./ sigma * (R1 - Q1*S1/G)   
    ### d_beta0_L3
    d_beta0_L3 = -1./sigma * beta0 * (S0 - S1*S1/G)    
    ###  d_beta0_prior_on_alpha 
    d_beta0_prior_on_alpha = 0    
    ###  d_beta0_prior_on_sigma
    d_beta0_prior_on_sigma = 0    
    
    
    
    dhsigma = -1./N *(d_sig_L0 + d_sig_L1 + d_sig_L2 + d_sig_L3 + d_sig_prior_on_alpha + d_sig_prior_on_sigma) 
    
    dhbeta0 = -1./N *(d_beta0_L0 + d_beta0_L1 + d_beta0_L2 + d_beta0_L3 + d_beta0_prior_on_alpha + d_beta0_prior_on_sigma) 
    
    if int_type ==0:
        return np.array((dhsigma, dhbeta0))
    else:
        ### d_alpha_L0
        d_alpha_L0 = - 0.5 * 1./alpha * S3/G + 0.5 * np.sum(genotype) * 1./alpha 
        ### d_alpha_L1
        d_alpha_L1 = -0.5 * 1./sigma * (1./alpha) *(O1 - 2*Q1*Q2/G + Q1*Q1*S3/(G*G))  
        ### d_alpha_L2
        d_alpha_L2 = 1./sigma * beta0 * 1./alpha * (Q1 - Q2 *S1/G - Q1*S2/G + Q1*S1*S3/(G*G))  
        ### d_alpha_L3
        d_alpha_L3 = -0.5* 1./sigma *beta0*beta0 *1./alpha * (S1 -  2*S1*S2/G + S1*S1*S3/(G*G))  
        
        d_alpha_prior_on_sigma = 0 
        
        dhalpha = -1./N *(d_alpha_L0 + d_alpha_L1 + d_alpha_L2 + d_alpha_L3 + d_alpha_prior_on_alpha + d_alpha_prior_on_sigma) 
        return np.array((dhalpha, dhsigma, dhbeta0))

####################################################################################################
def INLACauchy_hhprime(x,params):
    ''' evalute the second order derivative
    depends on: INLACauchy_intermediary_params
    '''
    [genotype, phenotype, theta1, theta2, v, lam, N, int_type,gamma] = params
    
    if int_type ==0:
        alpha = 1.
        sigma = x[0]
        beta0 = x[1]
        
    else:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        if int_type == -1: # won't get here for cauchy
            daa_prior_on_alpha = - (1./lam -1)*1./(alpha* alpha)
        else:
            T1 = gamma**2 * (gamma**2 - 2) + 2*(gamma**2 + 1)*(np.log(alpha)**2)
            T1 = T1 + 2 * gamma**2 * np.log(alpha)  + (np.log(alpha)**4) 
            T1 = T1 + 2. * (np.log(alpha)**3)
            T2 = alpha**2 * ((gamma**2 + np.log(alpha)**2)**2)
            daa_prior_on_alpha =  T1/T2 # <--for Cauchy
            
        
    [S, R, Q, O] = INLACauchy_intermediary_params(x, genotype, phenotype, int_type)
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    
    G = sigma * v + S2
    
    if G ==0.:
        #print('error at params, G, hprime')
        #print(params)
        G = 0.00000001
        
    if sigma ==0.:
        #print('error at params, sigma, hprime')
        #print(params)
        sigma = 0.00000001
    
    ################# dss #################
    
    ### dss_L0
    dss_L0 = -(-N/2. + 0.5)*1./(sigma**2) + 0.5 * (v**2)/(G**2)  
    
    ### dss_L1 
    dss_L1 = -1./(sigma**3)*(R2 - Q1*Q1/G) + 1.0/(sigma**2) * (v * (Q1**2)/(G**2)) +\
    1./sigma * (Q1**2) * (v**2)/(G**3)
    
    ### dss_L2
    dss_L2 = 2./(sigma**3) * beta0 * (R1 - Q1*S1/G) - beta0/(sigma**2)*Q1*S1*v/(G**2) +\
    (-1./(sigma**2)) * beta0 * Q1*S1*v/(G**2) + 1./sigma * beta0 * (-2) * Q1*S1*v*v/(G**3) 
    
    ### dss_L3  
    dss_L3 =   -1./(sigma**3) * (beta0**2) *(S0 - S1*S1/G) + 1./(sigma**2)*(beta0**2)*(S1**2)*v/(G**2)+\
    + 1./sigma * (beta0**2) * (S1**2)*v*v/(G**3)
    
    ### dss_prior_on_alpha
    dss_prior_on_alpha = 0  
    
    ### dss_prior_on_sigma
    dss_prior_on_sigma = -(-theta1 -1)*1./(sigma**2) - 2* theta2/ (sigma**3)  
    
    
    ################# dsb #################
    
    ### dsb_L0
    dsb_L0 = 0   
    ### dsb_L1
    dsb_L1 = 0  
    ### dsb_L2
    dsb_L2 = -1./(sigma**2) * (R1 - Q1*S1/G) +1./sigma * Q1*S1*v/(G**2) 
    ### dsb_L3
    dsb_L3 = beta0/(sigma**2) * (S0 - S1**2/G) - (beta0/sigma) * (S1**2)*v/(G**2)  
    ### dsb_prior_on_alpha
    dsb_prior_on_alpha = 0     
    ### dsb_prior_on_sigma
    dsb_prior_on_sigma = 0   
    
    ################# dbb #################
    
    ### dbb_L0
    dbb_L0 = 0   
    ### dbb_L1
    dbb_L1 = 0    
    ### dbb_L2
    dbb_L2 = 0   
    ### dbb_L3
    dbb_L3 = -1./(sigma) * (S0 - S1*S1/G) 
    ### dbb_prior_on_alpha
    dbb_prior_on_alpha = 0  
    
    ### dbb_prior_on_sigma
    dbb_prior_on_sigma = 0   
    
    
    hss = -1./N *(dss_L0 + dss_L1 + dss_L2 + dss_L3 + dss_prior_on_alpha + dss_prior_on_sigma)
    hsb = -1./N *(dsb_L0 + dsb_L1 + dsb_L2 + dsb_L3 + dsb_prior_on_alpha + dsb_prior_on_sigma)
    hbb = -1./N *(dbb_L0 + dbb_L1 + dbb_L2 + dbb_L3 + dbb_prior_on_alpha + dbb_prior_on_sigma)
    
    if int_type ==0:
        return np.array((hss,hsb,hsb,hbb))
    else:
        ################# daa #################
        ### daa_L0 
        daa_L0 = 1./(2*(alpha**2)) * 1./G *(S3 - S4) + 1./(2 * alpha**2) * (S3**2)/(G**2) - 1./(2* alpha**2)* np.sum(genotype) 
    
        ### daa_L1
        P1 = O1 - 2*Q1*Q2/G + (Q1**2)*S3/(G**2) 
        P2 = O2 - (2* Q2**2 + 2* Q1 * Q3)/G + (4.* Q1*Q2*S3 + (Q1**2)*S4)/(G**2) - 2.*(Q1**2)*(S3**2)/(G**3) 
        ##                 ##
        
        daa_L1 = 0.5 * 1./sigma * 1./(alpha**2) * P1 - 0.5 *1./sigma * 1./(alpha**2)* P2 
    
        ### daa_L2
        P1 = Q1 - (Q2*S1 + Q1*S2)/G + (Q1*S1*S3)/(G**2)
        P2 = Q2 - (Q3*S1 + Q2*S2 + Q2*S2 + Q1*S3)/G + (S3 * ( Q2*S1 + Q1*S2 ))/(G**2) +\
        (Q2*S1*S3 + Q1 * S2*S3 + Q1*S1*S4 )/(G**2) - (2*Q1*S1 * (S3**2))/(G**3) 
        
        daa_L2 = -1./sigma * beta0 * 1./(alpha**2) * P1 + 1./sigma * beta0 * 1./(alpha**2) * P2 
    
        ### daa_L3
        P1 = S1 - 2* S1*S2/G + (S1**2)*S3/(G**2)
        P2 = S2 - 2* (S2**2 + S1*S3)/G + 2 * S1 * S2 * S3/ (G**2) + (2*S1 *S2 * S3 + S1**2 * S4)/(G**2) - (2* S1**2 * S3**2)/(G**3)
        
        daa_L3 = 1./ (2 * sigma) * (beta0**2)/ (alpha**2) * P1 - 1./(2*sigma)* (beta0**2)/(alpha**2) * P2   
    
        ### daa_prior_on_alpha
        #daa_prior_on_alpha = ALREADY DEFINED
    
        ### daa_prior_on_sigma
        daa_prior_on_sigma = 0  
    
        ################# das #################
        ### das_L0
        das_L0 = 0.5 * 1./alpha * v * S3 / (G**2) 
    
        ### das_L1
        P1 = O1 - 2.* Q1 * Q2/G + (Q1**2) * S3 / (G**2)
        P2 = 2* Q1*Q2*v/(G**2) - 2 * Q1**2 * S3 * v/ (G**3) 
        
        das_L1 = 0.5 * (1./(sigma**2)) * (1./alpha) * P1 - (0.5 * 1./sigma)*(1./alpha)* P2  
    
        ### das_L2
        P1 = Q1 - Q2 * S1/G - Q1 * S2/G + Q1 * S1 * S3/ (G**2)
        P2 = v*(Q2 * S1 + Q1 * S2)/ (G**2) - 2 * (Q1 * S1 * S3 * v)/(G**3)  
        das_L2 = - beta0 / (alpha * (sigma**2)) * P1 + beta0/ (alpha * sigma) * P2    
    
        ### das_L3
        P1 = 1./alpha * (S1 - 2*S1*S2/G + S1**2 * S3 / (G**2)) 
        P2 = 1./alpha * (2 * S1 * S2 *v/ (G**2) - 2 * S3 * (S1**2) * v/(G**3)) 
        das_L3 = beta0**2/ (2 * sigma**2)* P1 - (beta0**2)/ (2*sigma) * P2  
    
        ### das_prior_on_alpha
        das_prior_on_alpha = 0  
    
        ### das_prior_on_sigma
        das_prior_on_sigma = 0  
    
        ################# dab #################
        ### dab_L0
        dab_L0 = 0  
    
        ### dab_L1
        dab_L1 = 0   
    
        ### dab_L2
        dab_L2 = 1./(sigma * alpha ) * (Q1 - Q2 * S1/ G - Q1 * S2 /G + Q1 * S1 * S3/ (G**2)) 
    
        ### dab_L3
        dab_L3 = - beta0/(sigma * alpha) * (S1 - 2 * S1 * S2/G + S1**2 * S3/(G**2))  
    
        ### dab_prior_on_alpha
        dab_prior_on_alpha = 0   
    
        ### dab_prior_on_sigma
        dab_prior_on_sigma = 0    
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        
        haa = -1./N *(daa_L0 + daa_L1 + daa_L2 + daa_L3 + daa_prior_on_alpha + daa_prior_on_sigma) 
        has = -1./N *(das_L0 + das_L1 + das_L2 + das_L3 + das_prior_on_alpha + das_prior_on_sigma) 
        hab = -1./N *(dab_L0 + dab_L1 + dab_L2 + dab_L3 + dab_prior_on_alpha + dab_prior_on_sigma) 
        
        hsa = has
        hba = hab
        hbs = hsb
        return  np.array((haa, has, hab, hsa, hss, hsb, hba, hbs, hbb))
    
####################################################################################################
def INLACauchy_h_hess(x,params):    
    hess_entries = INLACauchy_hhprime(x,params)
    
    if params[-1]==0:
        Hess = np.resize(hess_entries,(2,2)) 
    else:
        Hess = np.resize(hess_entries,(3,3))
    
    if (np.linalg.det(Hess)) == 0:
        print('det dim  is 0')
        #return 'Error'
        return 0.00000001
        
    big_sigma = np.linalg.det(Hess) 
    return big_sigma
####################################################################################################
def INLACauchy_intermediary_params(x, genotype, phenotype, int_type):
       
    if int_type!= 0:
        alpha = x[0]
        #sigma = x[1]
        #beta0 = x[2]
        
        [S0,S1,S2,S3,S4,R1,R2,R3,Q1,Q2,Q3,O1,O2,O3] = np.zeros(14)
        for i in range(len(genotype)):
            
            #print genotype[i]
            c = alpha**genotype[i]
            [S0,S1,S2,S3,S4] = [S0 + c, S1 + genotype[i] *c, S2 + (genotype[i]**2) *c, S3 + (genotype[i]**3) *c, S4 + (genotype[i]**4) *c]
            [R1, R2, R3] = [R1 + (phenotype[i]**1) *c, R2 + (phenotype[i]**2) *c, R3 + (phenotype[i]**3) *c]
            [Q1, Q2, Q3] = [Q1 + (phenotype[i]**1)*(genotype[i]**1)*c, Q2 + (phenotype[i]**1)*(genotype[i]**2)*c, Q3 + (phenotype[i]**1)*(genotype[i]**3)*c]
            [O1, O2, O3] = [O1 + (phenotype[i]**2)*(genotype[i]**1)*c, O2 + (phenotype[i]**2)*(genotype[i]**2)*c, O3 + (phenotype[i]**2)*(genotype[i]**3)*c]
        
    else:
        alpha = 1.
        #sigma = x[0]
        #beta0 = x[1]
        [S0,S1,S2,S3,S4,R1,R2,R3,Q1,Q2,Q3,O1,O2,O3] = np.zeros(14)
        for i in range(len(genotype)):
            c = 1
            [S0,S1,S2,S3,S4] = [S0 + c, S1 + genotype[i] *c, S2 + (genotype[i]**2) *c, S3 + (genotype[i]**3) *c, S4 + (genotype[i]**4) *c]
            [R1, R2, R3] = [R1 + (phenotype[i]**1) *c, R2 + (phenotype[i]**2) *c, R3 + (phenotype[i]**3) *c]
            [Q1, Q2, Q3] = [Q1 + (phenotype[i]**1)*(genotype[i]**1)*c, Q2 + (phenotype[i]**1)*(genotype[i]**2)*c, Q3 + (phenotype[i]**1)*(genotype[i]**3)*c]
            [O1, O2, O3] = [O1 + (phenotype[i]**2)*(genotype[i]**1)*c, O2 + (phenotype[i]**2)*(genotype[i]**2)*c, O3 + (phenotype[i]**2)*(genotype[i]**3)*c]
        

    S = [S0, S1, S2, S3, S4]
    R = [R1, R2, R3] 
    Q = [Q1, Q2, Q3] 
    O = [O1, O2, O3]
    return [S, R, Q, O]

def set_global_params():
    '''
    sets up the global hyperparameters
    '''
    global theta1  # parameter for conjugate prior
    global theta2  # parameter for conjugate prior
    global v # parameter for prior
    global lam # parameter for prior
    global b 
    global gamma # for Cauchy

    theta1 = 1.
    theta2 = 2.
    v  = 5.
    b = 10.
    lam = 5.
    M = 500
    gamma = 1. 
    return []

