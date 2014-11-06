#!/home/joe/pymc2Python/bin/python
""" This script simply suns the pyMC2 sampler """
import matplotlib.pyplot as plt
import pymc as pm
import SEDFitterModel_pyMC2

S = pm.MCMC(SEDFitterModel_pyMC2, db='pickle')
S.sample(iter=1000, burn=0, thin=1)
