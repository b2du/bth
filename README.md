# bth-
working scripts for heteroskedastic tests <br />
last updated: July, 2015 <br />


## simple example

from simulations import * <br />
from BTHsuite import * <br />

[maf, n, intercept, beta, sigma2, logalpha,T,c]  = [0.2, 300,0, 0.1, 1., 0.2, 1, 0 ] <br />
[genotype, phenotype] = sim_xy(maf, n, intercept, beta, sigma2, logalpha,T,c) <br />

!# compute Bayes Factors using BTH for the generated data <br />
logBF = BTH(genotype,phenotype) <br />
!# compute Bayes Factors for the shuffled data  <br />
[genotypePermuted, phenotypePermuted] = special_shuffle2(genotype,phenotype) <br />
logBFperm = BTH(genotypePermuted, phenotypePermuted) <br />
