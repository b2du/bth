########################################################################################
#######################     helper functions  for BTHsuite    ##########################
########################################################################################
#requierements
import numpy as np
import random
########################################################################################
#Summary of functions
# 1. generate_x: generates a genotype vector
# 2. generate_y: generates a phenotype vector (basic)
# 3. sim_xy: generates jointly pairs (genotype,phenotype) from particular transformations
# 4. special_shuffle: shuffles the phenotype corresponding to particular genotype values 
########################################################################################

# Typical simulations
def generate_x(maf, n):
    ''' generates n samples, with given MAF
    the result is a vector whose entries are 0,1 and 2
    '''
    x =  np.random.binomial(2,maf,n)
    while (len(np.where(x==0)[0])*len(np.where(x==1)[0])*len(np.where(x==2)[0])==0):
        x =  np.random.binomial(2,maf,n)
    return x
########################################################################################
def generate_y(theta1, theta2, alpha, beta, sigma, genotype,intercept):
    '''
    generate response variable which correspond to the genotypes generated in generate_x
    '''
    y = []
    for i in range(len(genotype)):
        
        err_i = np.random.normal(0,sigma * (alpha**(-genotype[i]))) 
        # in the notes it might be sigma^2 but i replaced it for simplicity in evaluating the integral
        y.append(genotype[i]*beta + err_i+intercept)
    return y
########################################################################################

#Helper Functions for simulations and for writing to file
def sim_xy(maf, n, intercept, beta, sigma2, logalpha,T,c):
    '''
    Generates pairs (x,y)
    @input
    
    maf = minor allele frequency
    n = size of the genotype and phenotype
    intercept
    beta = effect size
    sigma2 = fixed variance
    logalpha = heteroskedasticity
    T = code for data type + transformation
    c = continuous vs discrete

    @output
    x, y -- tuples of size n
    x_i = 0, 1 or 2 binomial of parameter maf ,for i = 1,n 
    y_i = T(x_i) with given parameters
    '''

    # generate x
    x =  np.random.binomial(2,maf,int(n))
    while (len(np.where(x==0)[0])*len(np.where(x==1)[0])*len(np.where(x==2)[0])==0):
        x =  np.random.binomial(2,maf,int(n))

    if c ==0:
        xx = list(x)
    else:
        xx = list(x)
        for i in range(len(x)):
	    if x[i]==0:
	        err = np.random.normal(0,0.1)
	        while (err < 0 or err >= 2):
		    err = np.random.normal(0, 0.1)
		xx[i] = xx[i] + err
	    elif x[i]==2:
		err = np.random.normal(0,0.1)
                while (err < 0 or err >= 2):
                    err = np.random.normal(0, 0.1)
                xx[i] = xx[i] - err
	    else:
		err = np.random.normal(0,0.1)
		while (err < -1 or err > 1):
		    err = np.random.normal(0.,0.1)
		xx[i] = xx[i] + err    

    # generate y
    y = []
    alpha = np.e**(logalpha)
    for i in range(int(n)):
        err_i = np.random.normal(0,sigma2 * (alpha**(-xx[i])))
        #  sigma^2 replaced for simplicity in evaluating the integral
        # basic
	if T == 1 or T == 10 or T == 11 or T == 12:
	    y.append(xx[i]*beta + err_i+intercept)
	# log gaussian
	elif T == 5 or T == 51 or T == 52 or T == 50:
	    y.append(np.e**(xx[i]*beta + err_i + intercept))
	# random effect
	elif T == 2 or T == 20 or T == 21 or T == 22:
            err_j = np.random.normal(0, sigma2 + alpha*xx[i])
	    y.append( xx[i]*beta + err_j +intercept)
	# exponential mean
	elif T == 3 or T == 30 or T == 31 or T == 32:
	    y.append(np.exp(xx[i]*beta + intercept) + err_i)
	# log gaussian residual
	elif T == 4 or T == 40 or T == 41 or T ==42:
	    y.append(intercept + beta * xx[i] + np.e**(err_i) )
	# gamma distributed
	elif T == 6 or T == 60 or T == 61 or T == 62:
	    mu = 1./(intercept + beta * xx[i])
	    y.append(np.random.gamma(mu,1.))
	elif T == 7 or T ==70 or T == 71 or T == 72:
	    mu = np.e**(intercept + beta * xx[i])
	    mu = np.min(1./mu, mu)
	    y.append(np.random.negative_binomial(1,mu))
	elif T == 8 or T == 80 or T == 81 or T == 82:
	    component1 = np.random.normal(10, 0.2)
	    y.append(0.4 * component1 + 0.6 * (xx[i]*beta + err_i+intercept) )
 

    ok = abs(min(0,min(y)))
    y0 = ok+ 0.000000001
	
    if (T % 10 == 0 and T >= 10):
	for i in range(len(y)):
            y[i] = np.log(y[i]+ y0)
			
    if (T % 10 == 1 and T >= 10):
        for i in range(len(y)):
            y[i] = np.sqrt(y[i]+ y0)
	
    if (T % 10 == 2 and T >= 10):
        for i in range(len(y)):
            y[i] = (y[i]+ y0)**(1./3)
                      

    return [xx,y]

########################################################################################
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