################################################
# DESCRIPTION: tests for the BTH test using:   #
################################################ 
# a) Sampling implementation with Laplace prior#
# b) INLA implementation with Laplace prior    #
# c) INLA implementation with Cauchy prior     #
# d) 2SLS (eLife)                              #
# e) Brown - Forsythe (0,1,2 only)             #
# f) Levene (0,1,2 only)                       #
################################################
# includes the function special_shuffle        #
# for computing nulls                          #
################################################
# Date July 29, 2015                           #
# biancad, gdarnell, bee  @ princeton.edu      #
################################################

# requierements
import numpy as np
from numpy.random import randn
from scipy import stats
from scipy import optimize
import math
import random
import sys 
import os

from scipy.stats import loglaplace
from scipy.stats import cauchy

# global hyperparams
global M, b, theta1, theta2, v, lam, logexp_param, snp, out, gamma
 

def set_global_params():
    '''
    sets up the global hyperparameters
    '''
    global theta1, theta2, v, lam, b, M, gamma 
    
    theta1 = 1.
    theta2 = 2.
    v  = 5.
    b = 10.
    lam = 5.
    M = 500
    gamma = 1. # for Cauchy
    return []



########## Cauchy Inla ##########
def INLACauchy_uniform_intercept(snp,out):                    
    '''
    inla approximation with Cauchy prior and uniform prior over the intercept
    '''
    set_global_params()
    [numerator1, beta0_est, beta_est, alpha_est, sigma_est] = INLACauchy_log_Laplace( snp,out,theta1, theta2, v, b, 1,gamma)
    [denominator, beta0_est_null, beta_est_null, alpha_est_null, sigma_est_null] = INLACauchy_log_Laplace(snp,out,theta1, theta2, v, b, 0,gamma) # double integrals

    return [(numerator1 - denominator)* np.log10(np.e), beta0_est, beta_est, alpha_est, sigma_est, beta0_est_null, beta_est_null,alpha_est_null, sigma_est_null]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# helper functions for Cauchy, redundant # TO DO: consolidate with INLA Laplace

def INLACauchy_log_Laplace(snp, out,theta1, theta2, v,lam, int_type,gamma):
    ''' \int \int \int exp int_type can be 1, -1, 0 '''

    N = len(snp)

    if int_type == 1:
        bound1 = 0.000001     # for Cauchy
        bound2 = None     # for Cauchy
    elif int_type == -1:
        bound1 = 1.     # does not matter for Cauchy
        bound2 = None   # does not matter for Cauchy
    else:
        bound1 = None
        bound2 = None
               
# MAP estimates 
    N = len(snp)
    params = [snp, out, theta1, theta2, v, lam, N, int_type,gamma]
    ##############################################################
    if int_type != 0:
        # perform triple integral estimation with prior over alpha
        if int_type == -1:
            ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1.,1.,0.], fprime= lambda x: INLACauchy_hprime(x,params), \
                                bounds=((bound1, bound2),(0.00001,None),(None,None)),\
                                epsilon =1e-4,disp=False)
        else:
            #ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1.,1.,0.], fprime= lambda x: INLACauchy_hprime(x,params), \
            #                    bounds=((bound1, bound2),(0.00001,None),(None,None)),\
            #                    epsilon =1e-4)
            ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1.,1.,0.], fprime= lambda x: INLACauchy_hprime(x,params), \
                                bounds=((bound1, bound2),(0.000001,None),(None,None)),\
                                epsilon =1e-4,disp=False)
        
        
        [alpha_hat, sigma_hat, beta0_hat] = ans[0]
        
        #print("alpha_hat:",ans[0][0]," sigma_hat: ",ans[0][1], " beta0_hat: ",ans[0][2])
        evaluate_h = INLACauchy_h([alpha_hat, sigma_hat, beta0_hat], params) 
        evaluate_hess = INLACauchy_h_hess([alpha_hat, sigma_hat, beta0_hat], params) # fing the values of the hessian terms at MAP  
        d = 3.
        
        S2 = sum([(snp[i]**2) * (alpha_hat ** snp[i]) for i in range(len(snp))])
        S1 = sum([snp[i] * (alpha_hat ** snp[i]) for i in range(len(snp))])
        Q1 = sum([snp[i] * out[i] * (alpha_hat ** snp[i]) for i in range(len(snp))]) 
        
        
        beta_hat = 1./(v + 1./sigma_hat * S2) * 1./sigma_hat *(Q1 - beta0_hat*S1)
        #print("beta_hat:",beta_hat)
        #print('^ numerators ^')
    else:
        # perform double integral estimation with alpha = 1 
        ans = optimize.fmin_tnc( lambda x: INLACauchy_h(x,params), [1., 0.000001], fprime= lambda x: INLACauchy_hprime(x,params), \
                                bounds=((0.000001, None),(None, None)),epsilon =1e-5,disp=False)
        #print("alpha =1, sigma*_hat, beta0*_hat")
        [sigma_hat,beta0_hat,] = ans[0]
        #print("alpha_hat:",1," sigma_hat: ",ans[0][0], " beta0_hat: ",ans[0][1])

        evaluate_h = INLACauchy_h([sigma_hat, beta0_hat], params) # find the value of the h function at the MAP estimates
        evaluate_hess = INLACauchy_h_hess([sigma_hat,beta0_hat],params) # fing the values of the hessian terms at MAP   
        d = 2.
        alpha_hat = 1.
        S2 = sum([(snp[i]**2) * (alpha_hat ** snp[i]) for i in range(len(snp))])
        S1 = sum([snp[i] * (alpha_hat ** snp[i]) for i in range(len(snp))])
        Q1 = sum([snp[i] * out[i] * (alpha_hat ** snp[i]) for i in range(len(snp))]) 
               
        beta_hat = 1./(v + 1./sigma_hat * S2) * 1./sigma_hat *(Q1 - beta0_hat*S1)
    

    log_laplace_term = (- N * evaluate_h) + d/2. * np.log(2*np.pi) - \
     0.5 * np.log(abs(evaluate_hess)) - d/2. *np.log(N) 
         
    return [log_laplace_term, beta0_hat, beta_hat, alpha_hat,sigma_hat]


def INLACauchy_h(x,params):
    '''
    returns the evaluation of the main function h, that enters the Laplace approx in the term exp(- N * h)
    '''

    [snp, out, theta1, theta2, v, lam, N,int_type,gamma] = params
        
    if int_type != 0:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        
        if int_type == -1: #will not matter for Cauchy
            prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (1./lam - 1.) * np.log(alpha)
            #prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (1./lam ) * np.log(alpha)
        else:
            #prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (-1./lam - 1.) * np.log(alpha)
            prior_on_alpha = - np.log(np.pi * gamma + np.pi / gamma * ((np.log(alpha))**2)) - np.log(alpha) # Cauchy
        
    else:
        alpha = 1.
        sigma = x[0] 
        beta0 = x[1]
        prior_on_alpha = 0.
      

        
    #terms that i need in the evaluation of the function
    ####################################################
    [S, R, Q, O] = INLACauchy_intermediary_params(x, snp, out,int_type)
    
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    ####################################################
    
    G = sigma * v + S2
    prior_on_sigma =  (-theta1 -1)*np.log(sigma) - theta2*1./sigma

    if G ==0.:
        #print('error at params, G, h')
        #print(params)
        G = 0.000001
        
    if sigma == 0.:
        #print('error at params, sigma, h')
        #print(params)
        sigma = 0.000001
    
    L0 = (-N/2. + 0.5) * np.log(sigma) - 0.5 * np.log(G) + 0.5 * np.sum(snp) * np.log(alpha)
    L1 =  -1./(2 * sigma) *(R2 - Q1*Q1/G)
    L2 = 1./sigma * beta0 *(R1 - Q1*S1/G)
    L3 = - 1./(2 * sigma) *beta0 * beta0 * (S0 - S1*S1/G)
     
    h = -1./N *(L0 + L1 + L2 + L3 + prior_on_alpha + prior_on_sigma)
    
    return h
    


def INLACauchy_hprime(x,params):
    ''' evaluations of the first derivatives
    '''
    [snp, out, theta1, theta2, v, lam, N, int_type,gamma] = params
    if int_type != 0:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        if int_type == -1:
            d_alpha_prior_on_alpha = (1./lam -1.)*1./alpha # not used for Cauchy
        else:
            #d_alpha_prior_on_alpha = (-1./lam -1)*1./alpha
            d_alpha_prior_on_alpha =  - (gamma**2 + (np.log(alpha))**2 + 2 * np.log(alpha))/( gamma**2 *alpha + alpha * (np.log(alpha))**2)#for Cauchy
    else:    
        alpha = 1.
        sigma = x[0]
        beta0 = x[1]

        
    [S, R, Q, O] = INLACauchy_intermediary_params(x, snp, out, int_type)
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    
    G = sigma * v + S2
    
    if G ==0.:
        #print('error at params, G, hprime')
        #print(params)
        G = 0.000001
        
    if sigma ==0.:
        #print('error at params, sigma, hprime')
        #print(params)
        sigma = 0.000001
        
    if alpha ==0:
        #print('error at alpha,hprime')
        #print(params)
        alpha = 0.000001
    

    ### d_sig_L0
    d_sig_L0 = (-0.5*N + 0.5) * 1./sigma - 0.5 * v /G     #OK
    ### d_sig_L1
    d_sig_L1 = 1./(2* sigma*sigma)*(R2 - Q1*Q1/G) - 1./(2*sigma) * v * Q1*Q1/(G*G)    #OK
    ### d_sig_L2
    d_sig_L2 = -1./(sigma*sigma) *beta0 *( R1 - Q1*S1/G) + 1./sigma * beta0 * Q1* S1*v/ (G*G)    #OK    
    ### d_sig_L3
    d_sig_L3 = 1./ (2* sigma*sigma) * beta0* beta0 *(S0 - S1*S1/G) - 1./(2*sigma) *beta0*beta0 *S1*S1*v/(G*G)  #OK
    ### d_sig_prior_on_alpha
    d_sig_prior_on_alpha = 0 #OK
    ### d_sig_prior_on_sigma
    d_sig_prior_on_sigma = (-theta1 -1) *(1./sigma) + theta2/(sigma*sigma)   #OK
    
    
    
    ### d_beta0_L0
    d_beta0_L0 = 0   #OK
    ### d_beta0_L1
    d_beta0_L1 = 0   #OK
    ### d_beta0_L2
    d_beta0_L2 = 1./ sigma * (R1 - Q1*S1/G)   #OK
    ### d_beta0_L3
    d_beta0_L3 = -1./sigma * beta0 * (S0 - S1*S1/G)    #OK
    ###  d_beta0_prior_on_alpha 
    d_beta0_prior_on_alpha = 0    #OK
    ###  d_beta0_prior_on_sigma
    d_beta0_prior_on_sigma = 0    #OK
    
    
    
    dhsigma = -1./N *(d_sig_L0 + d_sig_L1 + d_sig_L2 + d_sig_L3 + d_sig_prior_on_alpha + d_sig_prior_on_sigma) #OK
    
    dhbeta0 = -1./N *(d_beta0_L0 + d_beta0_L1 + d_beta0_L2 + d_beta0_L3 + d_beta0_prior_on_alpha + d_beta0_prior_on_sigma) 
    
    if int_type ==0:
        return np.array((dhsigma, dhbeta0))
    else:
        ### d_alpha_L0
        d_alpha_L0 = - 0.5 * 1./alpha * S3/G + 0.5 * np.sum(snp) * 1./alpha 
        ### d_alpha_L1
        d_alpha_L1 = -0.5 * 1./sigma * (1./alpha) *(O1 - 2*Q1*Q2/G + Q1*Q1*S3/(G*G))  
        ### d_alpha_L2
        d_alpha_L2 = 1./sigma * beta0 * 1./alpha * (Q1 - Q2 *S1/G - Q1*S2/G + Q1*S1*S3/(G*G))  
        ### d_alpha_L3
        d_alpha_L3 = -0.5* 1./sigma *beta0*beta0 *1./alpha * (S1 -  2*S1*S2/G + S1*S1*S3/(G*G))   
        
        ### d_alpha_prior_on_alpha
        ### d_alpha_prior_on_sigma
        d_alpha_prior_on_sigma = 0 
        
        dhalpha = -1./N *(d_alpha_L0 + d_alpha_L1 + d_alpha_L2 + d_alpha_L3 + d_alpha_prior_on_alpha + d_alpha_prior_on_sigma) 
        return np.array((dhalpha, dhsigma, dhbeta0))
 
def INLACauchy_hhprime(x,params):
    
    [snp, out, theta1, theta2, v, lam, N, int_type,gamma] = params
    
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
            #daa_prior_on_alpha = - (-1./lam -1)* 1./(alpha*alpha)
            T1 = gamma**2 * (gamma**2 - 2) + 2*(gamma**2 + 1)*(np.log(alpha)**2)
            T1 = T1 + 2 * gamma**2 * np.log(alpha)  + (np.log(alpha)**4) 
            T1 = T1 + 2. * (np.log(alpha)**3)
            T2 = alpha**2 * ((gamma**2 + np.log(alpha)**2)**2)
            daa_prior_on_alpha =  T1/T2# <--for Cauchy
            
        
    [S, R, Q, O] = INLACauchy_intermediary_params(x, snp, out, int_type)
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    
    G = sigma * v + S2
    
    if G ==0.:
        #print('error at params, G, hprime')
        #print(params)
        G = 0.000001
        
    if sigma ==0.:
        #print('error at params, sigma, hprime')
        #print(params)
        sigma = 0.000001
    
    ################# dss #################
    
    ### dss_L0
    dss_L0 = -(-N/2. + 0.5)*1./(sigma**2) + 0.5 * (v**2)/(G**2)    
    
    ### dss_L1 TYPO!!! 
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
        daa_L0 = 1./(2*(alpha**2)) * 1./G *(S3 - S4) + 1./(2 * alpha**2) * (S3**2)/(G**2) - 1./(2* alpha**2)* np.sum(snp) 
    
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
        # oldP2 = S2 - 2* (S2**2 + S1*S3)/G - 2 * S1 * S2 * S3/ (G**2) + (2*S1 *S2 * S3 + S1**2 * S4)/(G**2) - (2* S1**2 * S3**2)/(G**3)
        P2 = S2 - 2* (S2**2 + S1*S3)/G + 2 * S1 * S2 * S3/ (G**2) + (2*S1 *S2 * S3 + S1**2 * S4)/(G**2) - (2* S1**2 * S3**2)/(G**3)
        
        daa_L3 = 1./ (2 * sigma) * (beta0**2)/ (alpha**2) * P1 - 1./(2*sigma)* (beta0**2)/(alpha**2) * P2   
    
        ### daa_prior_on_alpha    
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
    

def INLACauchy_h_hess(x,params):    
    hess_entries = INLACauchy_hhprime(x,params)
    
    if params[-1]==0:
        Hess = np.resize(hess_entries,(2,2)) 
    else:
        Hess = np.resize(hess_entries,(3,3))
    
    if (np.linalg.det(Hess)) == 0:
        return 0.000001
        
    big_sigma = np.linalg.det(Hess) 
    return big_sigma

def INLACauchy_intermediary_params(x, snp, out, int_type):
       
    if int_type!= 0:
        alpha = x[0]
        #sigma = x[1]
        #beta0 = x[2]
        
        [S0,S1,S2,S3,S4,R1,R2,R3,Q1,Q2,Q3,O1,O2,O3] = np.zeros(14)
        for i in range(len(snp)):
            
            #print snp[i]
            c = alpha**snp[i]
            [S0,S1,S2,S3,S4] = [S0 + c, S1 + snp[i] *c, S2 + (snp[i]**2) *c, S3 + (snp[i]**3) *c, S4 + (snp[i]**4) *c]
            [R1, R2, R3] = [R1 + (out[i]**1) *c, R2 + (out[i]**2) *c, R3 + (out[i]**3) *c]
            [Q1, Q2, Q3] = [Q1 + (out[i]**1)*(snp[i]**1)*c, Q2 + (out[i]**1)*(snp[i]**2)*c, Q3 + (out[i]**1)*(snp[i]**3)*c]
            [O1, O2, O3] = [O1 + (out[i]**2)*(snp[i]**1)*c, O2 + (out[i]**2)*(snp[i]**2)*c, O3 + (out[i]**2)*(snp[i]**3)*c]
        
    else:
        alpha = 1.
        #sigma = x[0]
        #beta0 = x[1]
        [S0,S1,S2,S3,S4,R1,R2,R3,Q1,Q2,Q3,O1,O2,O3] = np.zeros(14)
        for i in range(len(snp)):
            c = 1
            [S0,S1,S2,S3,S4] = [S0 + c, S1 + snp[i] *c, S2 + (snp[i]**2) *c, S3 + (snp[i]**3) *c, S4 + (snp[i]**4) *c]
            [R1, R2, R3] = [R1 + (out[i]**1) *c, R2 + (out[i]**2) *c, R3 + (out[i]**3) *c]
            [Q1, Q2, Q3] = [Q1 + (out[i]**1)*(snp[i]**1)*c, Q2 + (out[i]**1)*(snp[i]**2)*c, Q3 + (out[i]**1)*(snp[i]**3)*c]
            [O1, O2, O3] = [O1 + (out[i]**2)*(snp[i]**1)*c, O2 + (out[i]**2)*(snp[i]**2)*c, O3 + (out[i]**2)*(snp[i]**3)*c]
        

    S = [S0, S1, S2, S3, S4]
    R = [R1, R2, R3] 
    Q = [Q1, Q2, Q3] 
    O = [O1, O2, O3]
    return [S, R, Q, O]

################################## Laplace Sample ######################################
def SampleLaplace_uniform_intercept(snp,out):
    '''
    sampling version of the BTH test
    '''
    from scipy.stats import loglaplace
    global theta1, theta2, v, lam, M
    set_global_params()
    
    M = 500

    #data is assumed to be tested for correctness
    #BF = 1/m \sum \frac{T(\alpha_m)}{T_1}    
    alphas = loglaplace.rvs(1./lam, size= M)
    #print('alpha')
    #print(alphas)
    Theta1 = theta1
    Theta2 = theta2
    V = v
    Lam = lam

    # replace the above with the average
    log10_T1 = log10_T_s(snp,out,Theta1, Theta2, V, Lam, M, 1)
    #BF_terms= [[log10_T_s(snp,out,Theta1, Theta2, V, Lam, m, alpha) - log10_T1,alpha] for alpha in alphas]
    #return [BF_terms,alphas] 
    BF_terms= [log10_T_s(snp,out,Theta1, Theta2, V, Lam, M, alpha) - log10_T1 for alpha in alphas]
    return max(BF_terms) - np.log10(M)


# helper functions for Sample Laplace
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def log10_T_s(snp,out,theta1, theta2,v,lam, M, alpha):
    ''' return the log factors representing the numerator and denominator
    '''
    
    N = len(snp)
    if alpha==1.:
        #ln alpha (0.5 * sum(snp) 
        res = 0.
    else:
        res = np.log(alpha) * (0.5 * np.sum(snp)) *np.log10(np.e)

    Xo_alpha = float(sum([(snp[i]**2) * (alpha**snp[i]) for i in range(N)]))
    Yo_alpha = float(sum([(out[i]**2) * (alpha**snp[i]) for i in range(N)]))
    Zo_alpha = float(sum([(snp[i]*out[i]) * (alpha**snp[i]) for i in range(N)]))
    To_alpha = float(sum([(alpha**snp[i]) for i in range(N)]))
    TXo_alpha = float(sum([(snp[i]) * (alpha**snp[i]) for i in range(N)]))
    TYo_alpha = float(sum([(out[i]) * (alpha**snp[i]) for i in range(N)]))
    
    
    # MAP estimates 
    params = [N, Xo_alpha, Yo_alpha, Zo_alpha, To_alpha, TXo_alpha, TYo_alpha]
    #print('alpha')
    #print(alpha)
    ans = optimize.fmin_tnc( lambda x: h_s(x,params), [0,1], fprime= lambda x: hprime_s(x,params), bounds=((None, None),(0.0001, None)),disp=False)
    [beta0_hat,sigma_hat] = ans[0]
    #print('estimates')
    #print(ans[0])
    ##############################################################
    evaluate_h = h_s([beta0_hat,sigma_hat], params) # find the value of the h function at the MAP estimates
    #print('h at MAP')
    #print(evaluate_h)
    evaluate_hess = h_hess_s([beta0_hat,sigma_hat],params) # fing the values of the hessian terms at MAP   
    #print('hess at MAP')
    #print(evaluate_hess)
    d = 2. # dimension of the laplace approximation

    laplace_term = (- N * evaluate_h)* np.log10(np.e) + d/2. * np.log10(2*np.pi) - \
     0.5 * np.log10(0.0000000000000000000001 + abs(evaluate_hess)) - d/2. *np.log10(N) # edit such that it becomes non zero

    return res + laplace_term

######################################
######################################

def h_s(x,params):
    '''
    the main function that enters the Laplace approx as exp(- N * h)
    '''

    [N, Xo_alpha, Yo_alpha, Zo_alpha, To_alpha, TXo_alpha, TYo_alpha] = params
    beta0 = x[0]
    sigma = x[1]



    [A, B, C, D, G, betaC, betaD] = intermediary_params_s(x,params)
    if G ==0.:
        print('error at params, G, h')
        print(params)

    if sigma == 0.:
        print('error at params, sigma, h')
        print(params)
    h = -1./N *( A * np.log(sigma) + 1./sigma * C + B * np.log(G) + D*1./G)
    return h

######################################
######################################

def hprime_s(x,params):
    ''' derivative
    '''
    global theta2
    global theta1
    global v
    [N, Xo_alpha, Yo_alpha, Zo_alpha, To_alpha, TXo_alpha, TYo_alpha] = params

    beta0 = x[0]
    sigma = x[1]

    [A, B, C, D, G, betaC, betaD] = intermediary_params_s(x,params)

    if G ==0.:
        print('error at params, G, hprime')
        print(params)

    if sigma ==0.:
        print('error at params, sigma, hprime')
        print(params)


    dhsigma = -1./N * (A/sigma - C/(sigma**2) + B*v/G - v* D/(G**2))
    dhbeta0 = -1./N * (1./sigma * betaC + 1./G * betaD)
    return np.array((dhbeta0, dhsigma))

######################################
######################################

def hhprime_s(x,params):
    global theta2
    global theta1
    global v
    [N, Xo_alpha, Yo_alpha, Zo_alpha, To_alpha, TXo_alpha, TYo_alpha] = params

    beta0 = x[0]
    sigma = x[1]

    [A, B, C, D, G, betaC, betaD] = intermediary_params_s(x,params)
    betaZ = - TXo_alpha
    betabetaY = 2 * To_alpha


    if Xo_alpha==0.:
        print('error at params, Xo_alpha, hhprime')
        print(params)
    betabetaC = -0.5*(betabetaY - 1./Xo_alpha * 2.* (betaZ**2))
    betabetaD = v* 1./Xo_alpha * (betaZ**2)

    if G ==0.:
        print('error at params, G,hhprime')
        print(params)

    if sigma ==0.:
        print('error at params, sigma, hhprime')
        print(params)
        

    hbb = -1./N * ( 1./sigma * betabetaC + 1./G * betabetaD)
    hsb = -1./N *( -1./(sigma**2) * betaC - v/(G**2) * betaD)
    hss = -1./N *(-1./(sigma**2) + 2* C/(sigma**3) - B*(v**2)/(G**2) + 2* D*(v**2)/(G**3))

    return  np.array((hbb, hsb, hsb, hss))


######################################
######################################


def h_hess_s(x,params):
    hess_entries = hhprime_s(x,params)

    Hess = np.resize(hess_entries,(2,2))
    if (np.linalg.det(Hess)) == 0:
        return 0.000001

    big_sigma = np.linalg.det(Hess)
    return big_sigma

######################################
######################################


def intermediary_params_s(x,params):
    global theta2
    global theta1
    global v
    [N, Xo_alpha, Yo_alpha, Zo_alpha, To_alpha, TXo_alpha, TYo_alpha] = params

    beta0 = x[0]
    sigma = x[1]

    A = -0.5 * (N-1) - theta1 - 1.
    B = -0.5
    Z_alpha = Zo_alpha - beta0 * TXo_alpha
    Y_alpha = Yo_alpha + beta0**2 * To_alpha - 2.* beta0* TYo_alpha

    if Xo_alpha==0:
        print('error at params')
        print(params)
    C =  - 0.5 *( Y_alpha - (Z_alpha**2)/Xo_alpha + 2. * theta2)
    D = - 0.5  * v * (Z_alpha**2)/Xo_alpha # TO this is repetitive with h, so make separate function

    betaY = 2.* beta0 * To_alpha - 2.* TYo_alpha
    betaZ = - TXo_alpha
    betaC = -0.5 *(betaY - 1./Xo_alpha * 2.* Z_alpha * betaZ)
    betaD = - v * Z_alpha * betaZ/Xo_alpha

    G = sigma *v + Xo_alpha

    return [A, B, C, D, G, betaC, betaD]

######################################
######################################

    

########################################################################################
################################## Laplace INLA ########################################
########################################################################################

def INLALaplace_uniform_intercept(snp,out):
                   
    '''
    integrated laplace for approximating BFs
    '''
    from scipy.stats import loglaplace
    global theta1, theta2, v, b
    set_global_params()
    
    [numerator1, beta0_est, beta_est, alpha_est, sigma_est] = INLA_log_Laplace( snp,out,theta1, theta2, v, b, 1)
    [numerator2, beta0_est_null, beta_est_null,alpha_est_null, sigma_est_null] = INLA_log_Laplace(snp, out,theta1, theta2, v, b, -1)
    
    denominator = INLA_log_Laplace(snp,out,theta1, theta2, v, b,0) # double integrals
    
    maxi = max(numerator1 - denominator, numerator2 - denominator)
    mini = min(numerator1 - denominator, numerator2 - denominator)
    if  maxi > 500:  
        return maxi * np.log10(np.e)
    elif mini < - 500:
        mini = -500

    return [np.log(np.e**maxi + np.e**mini) * np.log10(np.e), beta0_est, beta_est, alpha_est, sigma_est, beta0_est_null, beta_est_null,alpha_est_null, sigma_est_null]


# helper functions for Laplace INLA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def INLA_log_Laplace(snp, out,theta1, theta2, v,lam, int_type):
    ''' \int \int \int exp
    int_type can be 1, -1, 0 
    '''
    
    N = len(snp)
    
    if int_type == 1:
        bound1 = 0.
        bound2 = 1.     # edit july 8th, july 23
    elif int_type == -1:
        bound1 = 1.     # edit july 8th
        bound2 = None
    else:
        bound1 = None
        bound2 = None
               
# MAP estimates 
    N = len(snp)
    params = [snp, out, theta1, theta2, v, lam, N, int_type]
    ##############################################################
    if int_type != 0:
        # perform triple integral estimation with prior over alpha
        if int_type == -1:
            ans = optimize.fmin_tnc( lambda x: INLA_h(x,params), [1.,1.,0.], fprime= lambda x: INLA_hprime(x,params), \
                                bounds=((bound1, bound2),(0.00001,None),(None,None)),\
                                epsilon =1e-4,disp=False)
        else:
            ans = optimize.fmin_tnc( lambda x: INLA_h(x,params), [1.,1.,0.], fprime= lambda x: INLA_hprime(x,params), \
                                bounds=((bound1, bound2),(0.00001,None),(None,None)),\
                                epsilon =1e-4,disp=False)
        # edit july 20: changed start guess to 1.1 for alpha such that it is symmetric to other integral
        
        [alpha_hat, sigma_hat, beta0_hat] = ans[0]
        
        evaluate_h = INLA_h([alpha_hat, sigma_hat, beta0_hat], params) 
        evaluate_hess = INLA_h_hess([alpha_hat, sigma_hat, beta0_hat], params) # fing the values of the hessian terms at MAP  
        d = 3.
        
        S2 = sum([(snp[i]**2) * (alpha_hat ** snp[i]) for i in range(len(snp))])
        S1 = sum([snp[i] * (alpha_hat ** snp[i]) for i in range(len(snp))])
        Q1 = sum([snp[i] * out[i] * (alpha_hat ** snp[i]) for i in range(len(snp))]) 
        
        
        beta_hat = 1./(v + 1./sigma_hat * S2) * 1./sigma_hat *(Q1 - beta0_hat*S1)
        #print('^ numerators ^')
    else:
        # perform double integral estimation with alpha = 1 
        ans = optimize.fmin_tnc( lambda x: INLA_h(x,params), [1., 0.00001], fprime= lambda x: INLA_hprime(x,params), \
                                bounds=((0.00001, None),(None, None)),epsilon =1e-5,disp=False)
        #print("alpha =1, sigma*_hat, beta0*_hat")
        [sigma_hat,beta0_hat,] = ans[0]
        #print("alpha_hat:",1," sigma_hat: ",ans[0][0], " beta0_hat: ",ans[0][1])

        evaluate_h = INLA_h([sigma_hat, beta0_hat], params) # find the value of the h function at the MAP estimates
        evaluate_hess = INLA_h_hess([sigma_hat,beta0_hat],params) # fing the values of the hessian terms at MAP   
        d = 2.
        alpha_hat = 1.
        S2 = sum([(snp[i]**2) * (alpha_hat ** snp[i]) for i in range(len(snp))])
        S1 = sum([snp[i] * (alpha_hat ** snp[i]) for i in range(len(snp))])
        Q1 = sum([snp[i] * out[i] * (alpha_hat ** snp[i]) for i in range(len(snp))]) 
               
        beta_hat = 1./(v + 1./sigma_hat * S2) * 1./sigma_hat *(Q1 - beta0_hat*S1)
        #print("beta_hat:",beta_hat)
        #print('^denominator ^')
       
    
    log_laplace_term = (- N * evaluate_h) + d/2. * np.log(2*np.pi) - \
     0.5 * np.log(abs(evaluate_hess)) - d/2. *np.log(N) 
    return [log_laplace_term, beta0_hat, beta_hat,alpha_hat, sigma_hat]

################################################################################################
################################################################################################
def INLA_h(x,params):
    '''
    returns the evaluation of the main function h, that enters the Laplace approx in the term exp(- N * h)
    '''

    [snp, out, theta1, theta2, v, lam, N,int_type] = params
        
    if int_type != 0:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        
        if int_type == -1:
            prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (1./lam - 1.) * np.log(alpha)
            #prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (1./lam ) * np.log(alpha)
        else:
            prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (-1./lam - 1.) * np.log(alpha)
            #prior_on_alpha = np.log(0.5 * 1./abs(lam)) + (-1./lam ) * np.log(alpha)
        
    else:
        alpha = 1.
        sigma = x[0] 
        beta0 = x[1]
        prior_on_alpha = 0.
             
    #terms that i need in the evaluation of the function
    ####################################################
    [S, R, Q, O] = INLA_intermediary_params(x, snp, out,int_type)
    
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    ####################################################
    
    G = sigma * v + S2
    prior_on_sigma =  (-theta1 -1)*np.log(sigma) - theta2*1./sigma

    if G ==0.:
        #print('error at params, G, h')
        #print(params)
        G = 0.000001
        
    if sigma == 0.:
        #print('error at params, sigma, h')
        #print(params)
        sigma = 0.000001
    
    L0 = (-N/2. + 0.5) * np.log(sigma) - 0.5 * np.log(G) + 0.5 * np.sum(snp) * np.log(alpha)
    L1 =  -1./(2 * sigma) *(R2 - Q1*Q1/G)
    L2 = 1./sigma * beta0 *(R1 - Q1*S1/G)
    L3 = - 1./(2 * sigma) *beta0 * beta0 * (S0 - S1*S1/G)
     
    h = -1./N *(L0 + L1 + L2 + L3 + prior_on_alpha + prior_on_sigma)
    
    return h


################################################################################################
################################################################################################
def INLA_hprime(x,params):
    ''' evaluations of the first derivatives
    
    '''
    [snp, out, theta1, theta2, v, lam, N, int_type] = params
    if int_type != 0:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        if int_type == -1:
            d_alpha_prior_on_alpha = (1./lam -1.)*1./alpha
        else:
            d_alpha_prior_on_alpha = (-1./lam -1)*1./alpha
    else:    
        alpha = 1.
        sigma = x[0]
        beta0 = x[1]

        
    [S, R, Q, O] = INLA_intermediary_params(x, snp, out, int_type)
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    
    G = sigma * v + S2
    
    if G ==0.:
        #print('error at params, G, hprime')
        #print(params)
        G = 0.000001
        
    if sigma ==0.:
        #print('error at params, sigma, hprime')
        #print(params)
        sigma = 0.000001
    

    ### d_sig_L0
    d_sig_L0 = (-0.5*N + 0.5) * 1./sigma - 0.5 * v /G     #OK
    ### d_sig_L1
    d_sig_L1 = 1./(2* sigma*sigma)*(R2 - Q1*Q1/G) - 1./(2*sigma) * v * Q1*Q1/(G*G)    #OK
    ### d_sig_L2
    d_sig_L2 = -1./(sigma*sigma) *beta0 *( R1 - Q1*S1/G) + 1./sigma * beta0 * Q1* S1*v/ (G*G)    #OK    
    ### d_sig_L3
    d_sig_L3 = 1./ (2* sigma*sigma) * beta0* beta0 *(S0 - S1*S1/G) - 1./(2*sigma) *beta0*beta0 *S1*S1*v/(G*G)  #OK
    ### d_sig_prior_on_alpha
    d_sig_prior_on_alpha = 0 #OK
    ### d_sig_prior_on_sigma
    d_sig_prior_on_sigma = (-theta1 -1) *(1./sigma) + theta2/(sigma*sigma)   #OK
    
    
    
    ### d_beta0_L0
    d_beta0_L0 = 0   #OK
    ### d_beta0_L1
    d_beta0_L1 = 0   #OK
    ### d_beta0_L2
    d_beta0_L2 = 1./ sigma * (R1 - Q1*S1/G)   #OK
    ### d_beta0_L3
    d_beta0_L3 = -1./sigma * beta0 * (S0 - S1*S1/G)    #OK
    ###  d_beta0_prior_on_alpha 
    d_beta0_prior_on_alpha = 0    #OK
    ###  d_beta0_prior_on_sigma
    d_beta0_prior_on_sigma = 0    #OK
    
    dhsigma = -1./N *(d_sig_L0 + d_sig_L1 + d_sig_L2 + d_sig_L3 + d_sig_prior_on_alpha + d_sig_prior_on_sigma) #OK
    
    dhbeta0 = -1./N *(d_beta0_L0 + d_beta0_L1 + d_beta0_L2 + d_beta0_L3 + d_beta0_prior_on_alpha + d_beta0_prior_on_sigma) #OK
    
    if int_type ==0:
        return np.array((dhsigma, dhbeta0))
    else:
        ### d_alpha_L0
        d_alpha_L0 = - 0.5 * 1./alpha * S3/G + 0.5 * np.sum(snp) * 1./alpha #OK
        ### d_alpha_L1
        d_alpha_L1 = -0.5 * 1./sigma * (1./alpha) *(O1 - 2*Q1*Q2/G + Q1*Q1*S3/(G*G))  #OK
        ### d_alpha_L2
        d_alpha_L2 = 1./sigma * beta0 * 1./alpha * (Q1 - Q2 *S1/G - Q1*S2/G + Q1*S1*S3/(G*G))  #OK
        ### d_alpha_L3
        d_alpha_L3 = -0.5* 1./sigma *beta0*beta0 *1./alpha * (S1 -  2*S1*S2/G + S1*S1*S3/(G*G))   #OK
        ### d_alpha_prior_on_alpha
        # see before
        ### d_alpha_prior_on_sigma
        d_alpha_prior_on_sigma = 0 #OK
        
        dhalpha = -1./N *(d_alpha_L0 + d_alpha_L1 + d_alpha_L2 + d_alpha_L3 + d_alpha_prior_on_alpha + d_alpha_prior_on_sigma) 
        return np.array((dhalpha, dhsigma, dhbeta0))


################################################################################################
################################################################################################

def INLA_hhprime(x,params):
    
    [snp, out, theta1, theta2, v, lam, N, int_type] = params
    
    if int_type ==0:
        alpha = 1.
        sigma = x[0]
        beta0 = x[1]
        
    else:
        alpha = x[0]
        sigma = x[1]
        beta0 = x[2]
        if int_type == -1:
            daa_prior_on_alpha = - (1./lam -1)*1./(alpha* alpha)
            #daa_prior_on_alpha = - (1./lam )*1./(alpha* alpha)
        else:
            daa_prior_on_alpha = - (-1./lam -1)* 1./(alpha*alpha)
            #daa_prior_on_alpha = - (-1./lam )* 1./(alpha*alpha)
            
        
    [S, R, Q, O] = INLA_intermediary_params(x, snp, out, int_type)
    [S0, S1, S2, S3, S4] = S
    [R1, R2, R3] = R
    [Q1,Q2,Q3] = Q
    [O1,O2,O3] = O
    
    
    G = sigma * v + S2
    
    if G ==0.:
        #print('error at params, G, hprime')
        #print(params)
        G = 0.000001
        
    if sigma ==0.:
        #print('error at params, sigma, hprime')
        #print(params)
        sigma = 0.000001
    
    ################# dss #################
    
    ### dss_L0
    dss_L0 = -(-N/2. + 0.5)*1./(sigma**2) + 0.5 * (v**2)/(G**2)  #OK  
    
    ### dss_L1 TYPO!!! <---- July 15
    dss_L1 = -1./(sigma**3)*(R2 - Q1*Q1/G) + 1.0/(sigma**2) * (v * (Q1**2)/(G**2)) +\
    1./sigma * (Q1**2) * (v**2)/(G**3)
    
    ### dss_L2
    dss_L2 = 2./(sigma**3) * beta0 * (R1 - Q1*S1/G) - beta0/(sigma**2)*Q1*S1*v/(G**2) +\
    (-1./(sigma**2)) * beta0 * Q1*S1*v/(G**2) + 1./sigma * beta0 * (-2) * Q1*S1*v*v/(G**3) #OK
    
    ### dss_L3  #  TYPO  <------ july 15
    dss_L3 =   -1./(sigma**3) * (beta0**2) *(S0 - S1*S1/G) + 1./(sigma**2)*(beta0**2)*(S1**2)*v/(G**2)+\
    + 1./sigma * (beta0**2) * (S1**2)*v*v/(G**3)
    
    ### dss_prior_on_alpha
    dss_prior_on_alpha = 0  #OK
    
    ### dss_prior_on_sigma
    dss_prior_on_sigma = -(-theta1 -1)*1./(sigma**2) - 2* theta2/ (sigma**3)  #OK
    
    
    ################# dsb #################
    
    ### dsb_L0
    dsb_L0 = 0 #OK
    
    ### dsb_L1
    dsb_L1 = 0  #OK
    
    ### dsb_L2
    dsb_L2 = -1./(sigma**2) * (R1 - Q1*S1/G) +1./sigma * Q1*S1*v/(G**2) #OK
    
    ### dsb_L3
    dsb_L3 = beta0/(sigma**2) * (S0 - S1**2/G) - (beta0/sigma) * (S1**2)*v/(G**2)  #OK
    
    ### dsb_prior_on_alpha
    dsb_prior_on_alpha = 0  #OK
    
    ### dsb_prior_on_sigma
    dsb_prior_on_sigma = 0   #OK
    
    ################# dbb #################
    
    ### dbb_L0
    dbb_L0 = 0   #OK
    
    ### dbb_L1
    dbb_L1 = 0   #OK
    
    ### dbb_L2
    dbb_L2 = 0   #OK
    
    ### dbb_L3
    dbb_L3 = -1./(sigma) * (S0 - S1*S1/G) #OK
    
    ### dbb_prior_on_alpha
    dbb_prior_on_alpha = 0  #OK
    
    ### dbb_prior_on_sigma
    dbb_prior_on_sigma = 0   #OK
    
    
    hss = -1./N *(dss_L0 + dss_L1 + dss_L2 + dss_L3 + dss_prior_on_alpha + dss_prior_on_sigma)
    hsb = -1./N *(dsb_L0 + dsb_L1 + dsb_L2 + dsb_L3 + dsb_prior_on_alpha + dsb_prior_on_sigma)
    hbb = -1./N *(dbb_L0 + dbb_L1 + dbb_L2 + dbb_L3 + dbb_prior_on_alpha + dbb_prior_on_sigma)
    
    if int_type ==0:
        return np.array((hss,hsb,hsb,hbb))
    else:
        ################# daa #################
        ### daa_L0 Typo!!! - July 15 
        daa_L0 = 1./(2*(alpha**2)) * 1./G *(S3 - S4) + 1./(2 * alpha**2) * (S3**2)/(G**2) - 1./(2* alpha**2)* np.sum(snp) 
    
        ### daa_L1
        
        ## EDIT on July 13 ##
        P1 = O1 - 2*Q1*Q2/G + (Q1**2)*S3/(G**2) 
        P2 = O2 - (2* Q2**2 + 2* Q1 * Q3)/G + (4.* Q1*Q2*S3 + (Q1**2)*S4)/(G**2) - 2.*(Q1**2)*(S3**2)/(G**3) 
        ##                 ##
        
        daa_L1 = 0.5 * 1./sigma * 1./(alpha**2) * P1 - 0.5 *1./sigma * 1./(alpha**2)* P2 #OK
    
        ### daa_L2
        P1 = Q1 - (Q2*S1 + Q1*S2)/G + (Q1*S1*S3)/(G**2)
        P2 = Q2 - (Q3*S1 + Q2*S2 + Q2*S2 + Q1*S3)/G + (S3 * ( Q2*S1 + Q1*S2 ))/(G**2) +\
        (Q2*S1*S3 + Q1 * S2*S3 + Q1*S1*S4 )/(G**2) - (2*Q1*S1 * (S3**2))/(G**3) 
        
        daa_L2 = -1./sigma * beta0 * 1./(alpha**2) * P1 + 1./sigma * beta0 * 1./(alpha**2) * P2 #OK
    
        ### daa_L3
        P1 = S1 - 2* S1*S2/G + (S1**2)*S3/(G**2)
        # oldP2 = S2 - 2* (S2**2 + S1*S3)/G - 2 * S1 * S2 * S3/ (G**2) + (2*S1 *S2 * S3 + S1**2 * S4)/(G**2) - (2* S1**2 * S3**2)/(G**3)
        #EDIT july 7,3:18pm:
        P2 = S2 - 2* (S2**2 + S1*S3)/G + 2 * S1 * S2 * S3/ (G**2) + (2*S1 *S2 * S3 + S1**2 * S4)/(G**2) - (2* S1**2 * S3**2)/(G**3)
        
        daa_L3 = 1./ (2 * sigma) * (beta0**2)/ (alpha**2) * P1 - 1./(2*sigma)* (beta0**2)/(alpha**2) * P2   #OK
    
        ### daa_prior_on_alpha
        #daa_prior_on_alpha = ALREADY DEFINED
    
        ### daa_prior_on_sigma
        daa_prior_on_sigma = 0  #OK
    
        ################# das #################
        ### das_L0
        das_L0 = 0.5 * 1./alpha * v * S3 / (G**2) #OK
    
        ### das_L1
        P1 = O1 - 2.* Q1 * Q2/G + (Q1**2) * S3 / (G**2)
        P2 = 2* Q1*Q2*v/(G**2) - 2 * Q1**2 * S3 * v/ (G**3) # EDIT JULY 6th G**2 --> G**3 in last term  #OK
        
        das_L1 = 0.5 * (1./(sigma**2)) * (1./alpha) * P1 - (0.5 * 1./sigma)*(1./alpha)* P2  #OK
    
        ### das_L2
        P1 = Q1 - Q2 * S1/G - Q1 * S2/G + Q1 * S1 * S3/ (G**2)
        P2 = v*(Q2 * S1 + Q1 * S2)/ (G**2) - 2 * (Q1 * S1 * S3 * v)/(G**3)  # EDIT JULY 6th G**2 --> G**3 in last term
        das_L2 = - beta0 / (alpha * (sigma**2)) * P1 + beta0/ (alpha * sigma) * P2    #OK
    
        ### das_L3
        P1 = 1./alpha * (S1 - 2*S1*S2/G + S1**2 * S3 / (G**2)) #OK
        P2 = 1./alpha * (2 * S1 * S2 *v/ (G**2) - 2 * S3 * (S1**2) * v/(G**3)) #OK
        das_L3 = beta0**2/ (2 * sigma**2)* P1 - (beta0**2)/ (2*sigma) * P2  #OK
    
        ### das_prior_on_alpha
        das_prior_on_alpha = 0  #OK
    
        ### das_prior_on_sigma
        das_prior_on_sigma = 0  #OK
    
        ################# dab #################
        ### dab_L0
        dab_L0 = 0  #OK
    
        ### dab_L1
        dab_L1 = 0   #OK
    
        ### dab_L2
        dab_L2 = 1./(sigma * alpha ) * (Q1 - Q2 * S1/ G - Q1 * S2 /G + Q1 * S1 * S3/ (G**2)) #OK
    
        ### dab_L3
        dab_L3 = - beta0/(sigma * alpha) * (S1 - 2 * S1 * S2/G + S1**2 * S3/(G**2))  #OK
    
        ### dab_prior_on_alpha
        dab_prior_on_alpha = 0   #OK
    
        ### dab_prior_on_sigma
        dab_prior_on_sigma = 0    #OK
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        
        haa = -1./N *(daa_L0 + daa_L1 + daa_L2 + daa_L3 + daa_prior_on_alpha + daa_prior_on_sigma) 
        has = -1./N *(das_L0 + das_L1 + das_L2 + das_L3 + das_prior_on_alpha + das_prior_on_sigma) 
        hab = -1./N *(dab_L0 + dab_L1 + dab_L2 + dab_L3 + dab_prior_on_alpha + dab_prior_on_sigma) 
        
        hsa = has
        hba = hab
        hbs = hsb
        return  np.array((haa, has, hab, hsa, hss, hsb, hba, hbs, hbb))
    
################################################################################################
################################################################################################
def INLA_h_hess(x,params):    
    hess_entries = INLA_hhprime(x,params)
    
    if params[-1]==0:
        Hess = np.resize(hess_entries,(2,2)) 
    else:
        Hess = np.resize(hess_entries,(3,3))
    
    if (np.linalg.det(Hess)) == 0:
        return 0.000001
        
    big_sigma = np.linalg.det(Hess) 
    return big_sigma

################################################################################################
################################################################################################

def INLA_intermediary_params(x, snp, out, int_type):
       
    if int_type!= 0:
        alpha = x[0]
        #sigma = x[1]
        #beta0 = x[2]
        
        [S0,S1,S2,S3,S4,R1,R2,R3,Q1,Q2,Q3,O1,O2,O3] = np.zeros(14)
        for i in range(len(snp)):
            
            #print snp[i]
            c = alpha**snp[i]
            [S0,S1,S2,S3,S4] = [S0 + c, S1 + snp[i] *c, S2 + (snp[i]**2) *c, S3 + (snp[i]**3) *c, S4 + (snp[i]**4) *c]
            [R1, R2, R3] = [R1 + (out[i]**1) *c, R2 + (out[i]**2) *c, R3 + (out[i]**3) *c]
            [Q1, Q2, Q3] = [Q1 + (out[i]**1)*(snp[i]**1)*c, Q2 + (out[i]**1)*(snp[i]**2)*c, Q3 + (out[i]**1)*(snp[i]**3)*c]
            [O1, O2, O3] = [O1 + (out[i]**2)*(snp[i]**1)*c, O2 + (out[i]**2)*(snp[i]**2)*c, O3 + (out[i]**2)*(snp[i]**3)*c]
        
    else:
        alpha = 1.
        #sigma = x[0]
        #beta0 = x[1]
        [S0,S1,S2,S3,S4,R1,R2,R3,Q1,Q2,Q3,O1,O2,O3] = np.zeros(14)
        for i in range(len(snp)):
            c = 1
            [S0,S1,S2,S3,S4] = [S0 + c, S1 + snp[i] *c, S2 + (snp[i]**2) *c, S3 + (snp[i]**3) *c, S4 + (snp[i]**4) *c]
            [R1, R2, R3] = [R1 + (out[i]**1) *c, R2 + (out[i]**2) *c, R3 + (out[i]**3) *c]
            [Q1, Q2, Q3] = [Q1 + (out[i]**1)*(snp[i]**1)*c, Q2 + (out[i]**1)*(snp[i]**2)*c, Q3 + (out[i]**1)*(snp[i]**3)*c]
            [O1, O2, O3] = [O1 + (out[i]**2)*(snp[i]**1)*c, O2 + (out[i]**2)*(snp[i]**2)*c, O3 + (out[i]**2)*(snp[i]**3)*c]
        
        
    # Capping values, not sure if correct! check if it makes sense
    ##S = capmax([S0, S1, S2, S3, S4])
    ##R = capmax([R1, R2, R3]) 
    ##Q = capmax([Q1, Q2, Q3]) 
    ##O = capmax([O1, O2, O3])
    S = [S0, S1, S2, S3, S4]
    R = [R1, R2, R3] 
    Q = [Q1, Q2, Q3] 
    O = [O1, O2, O3]
    return [S, R, Q, O]


########################################################################################
##################################  2SLS eLife  #########################################
########################################################################################
def SLS_test(snp,out):

   # implements the test found in eLife publications, Brown et al, Durbin last
   x = np.array(snp)
   y = np.array(out)
   A = np.array([ x, np.ones(len(x))])
   w = np.linalg.lstsq(A.T,y)[0]
   residues = y - w[0]*x - w[1]
   residues = residues**2
   [test_stat, pvalue] = stats.spearmanr(residues,x)
   return abs(test_stat)
    
    

########################################################################################
##################################  Brown Forsythe  ####################################
########################################################################################

def BrFor(x, y):
    '''
    assumes 0 1 and 2 coding 
    '''
    sample0 = [y[i] for i in range(len(x)) if int(x[i])==0]  # y0
    sample1 = [y[i] for i in range(len(x)) if int(x[i])==1]  # y1
    sample2 = [y[i] for i in range(len(x)) if int(x[i])!=0 and int(x[i])!=1] #y2
    
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
##################################      Levene      ####################################
########################################################################################
    

 
########################################################################################
###############################  helper functions  #####################################
########################################################################################

# FUNCTIONS FOR GENERATING SNPS
# Typical simulations
def generate_x(maf, n):
    ''' generates n samples, with given MAF
    the result is a vector whose entries are 0,1 and 2
    '''
    x =  np.random.binomial(2,maf,n)
    while (len(np.where(x==0)[0])*len(np.where(x==1)[0])*len(np.where(x==2)[0])==0):
        x =  np.random.binomial(2,maf,n)
    return x

def generate_y(theta1, theta2, alpha, beta, sigma, snp,intercept):
    #beta = np.random.normal(0,1./v)
    #sigma = 1./ np.random.gamma(theta1,theta2) # possible source of error, double check

    '''
    generate response variable which correspond to the snps generated in generate_x
    '''
    y = []
    for i in range(len(snp)):
        
        err_i = np.random.normal(0,sigma * (alpha**(-snp[i]))) 
        # in the notes it might be sigma^2 but i replaced it for simplicity in evaluating the integral
        y.append(snp[i]*beta + err_i+intercept)
    return y


#Helper Functions for simulations and for writing to file
def sim_xy(maf, n, intercept, beta, sigma2, logalpha):
    '''
    x_i = 0, 1 or 2 binomial of parameter maf ,for i = 1,n 
    y_i = intercept + beta * x_i + N(0, sigma2 * alpha ^(- x_i))   
    @input
    maf
    n
    intercept
    beta
    sigma2
    logalpha

    @output
    x, y -- tuples of size n
    '''

    # generate x
    x =  np.random.binomial(2,maf,int(n))
    while (len(np.where(x==0)[0])*len(np.where(x==1)[0])*len(np.where(x==2)[0])==0):
        x =  np.random.binomial(2,maf,int(n))

    # generate y
    y = []
    alpha = np.e**(logalpha)
    for i in range(int(n)):
        err_i = np.random.normal(0,sigma2 * (alpha**(-x[i])))
        # in the notes it might be sigma^2 but i replaced it for simplicity in evaluating the integral
        y.append(x[i]*beta + err_i+intercept)

    return [x,y]

def sim_xyztv_to_file(x,y,z,t,v, output_name):
    '''
    write x and y  to a folder x_i \t y_i
    @input
    x, y are n tupples
    
    @output
    output_name with x,y values in current_folder/sims_1000/
    n rows, ith row contains x_i \t y_i
    '''

    with open(output_name,'w+') as f:
        for xx, yy,zz,tt,vv in zip(x, y,z,t,v):
            f.write(str(xx) + '\t'+str(yy) + '\t'+str(zz) + '\t'+ str(tt) + '\t'+str(vv) +'\n')

	return 0

def special_shuffle(genotype,phenotype):
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

def write_result(file_ptr,permuted,snp_id,snp_pos,maf,test_name,result):
    file_ptr.write(str(permuted) + " " + str(snp_id) + " " + str(snp_pos) + " " + str(maf) + " " +
          str(test_name) + " " + str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " +
                   str(result[3]) + " " + str(result[4]) + " " + str(result[5]) + " " + str(result[6]) +
                   " " + str(result[7]) + " " + str(result[8]) + "\n")
    file_ptr.flush()

def write_header(file_ptr):
    file_ptr.write("Permuted SNP_ID SNP_Pos MAF Test_Name Test_Stat Beta0 Beta Alpha Sigma Beta0_Null Beta_Null Alpha_Null Sigma_Null\n")
    file_ptr.flush()

def vedatest(snps,cis_pos,cis_IDs,phenotype,tests,results_filename):
    results_file = open(results_filename,"w")
    write_header(results_file)
    cis_pos = np.squeeze(np.asarray(cis_pos))
    cis_IDs = np.squeeze(np.asarray(cis_IDs))

    for i in range(snps.shape[0]):
        x = np.hstack(snps[i,])
        y = np.squeeze(np.asarray(phenotype))
        x = np.squeeze(np.asarray(x))
        
        maf = np.mean(x) / 2
        if(maf > 0.5):
            maf = 1 - maf
    
        if x.shape == y.shape and np.var(x) > 0:
            permuted = "0"
            if("INLACauchy" in tests):
                result = INLACauchy_uniform_intercept(x,y)
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"INLACauchy",result)
            if("INLALaplace" in tests):
                result = INLALaplace_uniform_intercept(x,y)
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"INLALaplace",result)
            if("SampleLaplace" in tests):
                result = SampleLaplace_uniform_intercept(x,y)
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"SampleLaplace",result)
            if("2SLS" in tests):
                result = SLS_test(x,y)
                result = (result,"-9","-9","-9","-9","-9","-9","-9","-9")
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"2SLS",result)

            permuted = "1"
            y = special_shuffle(x,y)[1]
            if("INLACauchy" in tests):
                result = INLACauchy_uniform_intercept(x,y)
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"INLACauchy",result)
            if("INLALaplace" in tests):
                result = INLALaplace_uniform_intercept(x,y)
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"INLALaplace",result)
            if("SampleLaplace" in tests):
                result = SampleLaplace_uniform_intercept(x,y)
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"SampleLaplace",result)
            if("2SLS" in tests):
                result = SLS_test(x,y)
                result = (result,"-9","-9","-9","-9","-9","-9","-9","-9")
                write_result(results_file,permuted,cis_IDs[i],cis_pos[i],maf,"2SLS",result)
    results_file.close()
