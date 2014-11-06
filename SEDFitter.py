#!/usr/bin/env python
""" This script is to FIT the SED a Heirachihal Bayesian Model of multi epoch data using Mone Carlo"""
import numpy as np
import pymc as pm
import theano
import theano.tensor as t
import scipy.interpolate
from DataModel import DataModel

FluxBands = ['BV','BR','BI','2MASSJ','2MASSH','2MASSK']
Appertures = [23,23,23,23,23,23]

SEDModel = DataModel(FluxBands,Appertures)  #Load the SED model

# Load observed flux and corresponding errors
ObservedData = np.genfromtxt('ObservedData.csv',delimiter=',',filling_values=None)
#ObservedData = np.ma.masked_equal(ObservedData, value=None)
ObservedDataSigma = np.genfromtxt('ObservedDataSigma.csv',delimiter=',',filling_values=None)
#ObservedDataSigma = np.ma.masked_equal(ObservedDataSigma, value=None)
T_nu = 3.0  # nu of the T distribution used to model observation. Should be > 2
T_lam = ObservedDataSigma *np.sqrt((T_nu -2.0)/T_nu)

print('Observed: ',ObservedData)
print('ObsError: ',ObservedDataSigma)

@pm.theano.compile.ops.as_op(itypes=[t.dscalar]*15,otypes=[t.dvector])
def FluxFromTheory(lTs,lMs,ldMenvbyMs,lRenv,ThetaCav,lMdbyMs,lRdo,lRdi,Zdisc,Bdisc,lalphad,lroCav,lroAmp,Inc,Av):
    """ Retunrs the expected flux table from Astrophysics model """
    GridParams = np.array((lTs,lMs,ldMenvbyMs,lRenv,ThetaCav,lMdbyMs,lRdo,lRdi,Zdisc,Bdisc,lalphad,lroCav,lroAmp,Inc))
    i, Flux = SEDModel.GetValue(GridParams,Av)
    return Flux

def GetUpperLowerBounds(param):
    """ Returns the Theano function to calculate the upper and lower bounds """
    @pm.theano.compile.ops.as_op(itypes=[t.dscalar],otypes=[t.dscalar,t.dscalar])
    def TheanoGetUpperLowerBounds(x):
        """ Returns the upper and lower bounds """
        return scipy.interpolate.splev(x,SEDModel.MaxMinSplrep[param][0]),scipy.interpolate.splev(x,SEDModel.MaxMinSplrep[param][1])

    return TheanoGetUpperLowerBounds

# Compile and keep the Theano functions for each parameter
GetUpperLowerBoundsFdic = dict()
for param in ['ldMenvbyMs','ThetaCav','lMdbyMs','lRdo','lRdi','Zdisc','Bdisc','lroAmp','lroCav'] :
    GetUpperLowerBoundsFdic[param] = GetUpperLowerBounds(param)


def ConvertRealToScaled(param,value):
    """ Converts the real value to scaled value which is used in montecarlo model """
    if SEDModel.MinMaxLimitsLog[param]['log']:
        value = np.log10(value)
    return (value-SEDModel.MinMaxLimitsLog[param]['min'])/(SEDModel.MinMaxLimitsLog[param]['max']-SEDModel.MinMaxLimitsLog[param]['min'])
    

@pm.theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar],otypes=[t.dscalar,t.dscalar])
def GetRenvUpperLowerBounds(age,mass):
    """ Returns the upper and lower bounds of log(Renv) """
    StarRad, StarTemp = SEDModel.GetRadTemp(age,mass)
    Rnot = (StarTemp/30.0)**2.5 * StarRad /2
    lbound = max(0.9*10**3,Rnot/4.0)  # 0.9 factor to take into considereaciton of no envelope models
    ubound = max(min(10**5,4*Rnot),10**3)
    #Convert to our scaled numbers between 1 and 0
    return ConvertRealToScaled('lRenv',ubound), ConvertRealToScaled('lRenv',lbound)
    

#paramindex = theano.shared(1)

with pm.Model() as model:
    
    #Priors
    lTs = pm.Uniform('lTs',lower=0,upper =1)  # Log(time_star)  lower=3,upper =7
    lMs = pm.Uniform('lMs',lower=0,upper= 1) # Log(M_star)  lower=0.1,upper= 50
#    paramindex.set_value(SEDModel.ParamsList.index('ldMenvbyMs'))
    upperMenv,lowerMenv = GetUpperLowerBoundsFdic['ldMenvbyMs'](lTs)
    ldMenvbyMs = pm.Uniform('ldMenvbyMs',upper= upperMenv, lower= lowerMenv)  # Log( d(M_env)/dt /M_star)
    upperRenv,lowerRenv = GetRenvUpperLowerBounds(lTs,lMs)
    lRenv = pm.Uniform('lRenv',upper= upperRenv,lower= lowerRenv)  # Log(R_env)
    upperThetaCav,lowerThetaCav = GetUpperLowerBoundsFdic['ThetaCav'](lTs)
    ThetaCav = pm.Uniform('ThetaCav',upper= upperThetaCav, lower= lowerThetaCav)  # Theta of Cavity
    upperlMdbyMs,lowerlMdbyMs = GetUpperLowerBoundsFdic['lMdbyMs'](lTs)
    lMdbyMs = pm.Uniform('lMdbyMs',upper= upperlMdbyMs, lower= lowerlMdbyMs)  # Log( M_disc /M_star)
    upperlRdo,lowerlRdo = GetUpperLowerBoundsFdic['lRdo'](lTs)
    lRdo = pm.Uniform('lRdo',upper= upperlRdo, lower= lowerlRdo)  # Log(R_disc_out)
    upperlRdi,lowerlRdi = GetUpperLowerBoundsFdic['lRdi'](lTs)# ALERT: Upper limit should be outer radius. Implement later
    lRdi = pm.Uniform('lRdi',upper= upperlRdi, lower= lowerlRdi)  # Log(R_disc_in)
    _,lowerZdisc = GetUpperLowerBoundsFdic['Zdisc'](lRdo)  # Upperlimit always 1
    Zdisc = pm.Uniform('Zdisc',upper= 1, lower= lowerZdisc)  # Z_disc
    upperBdisc, _ = GetUpperLowerBoundsFdic['Bdisc'](lRdo)  # Lowerlimit always 0
    Bdisc = pm.Uniform('Bdisc',upper= upperBdisc, lower= 0)  # Beta_disc
    lalphad = pm.Uniform('lalphad',lower=0,upper= 1) # Log(alpha_disc)  lower=0.001,upper= 0.1
    upperlroAmp,lowerlroAmp = GetUpperLowerBoundsFdic['lroAmp'](lMs)
    lroAmp = pm.Uniform('lroAmp',upper= upperlroAmp, lower= lowerlroAmp)  # Log( Ambient density)
    upperlroCav,lowerlroCav = GetUpperLowerBoundsFdic['lroCav'](lTs)# ALERT: Upper limit should be Ambient density. Implement later
    lroCav = pm.Uniform('lroCav',upper= upperlroCav, lower= lowerlroCav)  # Log(Envelope Cavity density)
    Inc = pm.Uniform('Inc',lower=0,upper= 1) # Inclination  lower=18.19,upper= 87.13

    distance = pm.Uniform('distance',lower=0.8,upper= 1.1) # Distance is between 0.8 and 1 kpc #Unit is kpc for Robitaille's flux
    Av = pm.Uniform('Av',lower=1,upper= 5) # Interstellar Av is between 1 and 5

    ### Now we calculate the expected flux from SED Model
    y_hat = FluxFromTheory(lTs,lMs,ldMenvbyMs,lRenv,ThetaCav,lMdbyMs,lRdo,lRdi,Zdisc,Bdisc,lalphad,lroCav,lroAmp,Inc,Av)/distance**2

    #Data likelihood
    y_like = pm.Normal('y_like',mu= y_hat, sd=ObservedDataSigma, observed=ObservedData)
#    y_like = pm.T('y_like',mu= y_hat, nu=T_nu, lam=T_lam, observed=ObservedData)  # T distribution for robustness

    # Inference...
#    start = pm.find_MAP() # Find starting value by optimization
    start = {'lTs':0.5,'lMs':0.5,'ldMenvbyMs':0.8,'lRenv':0.5,'ThetaCav':0.5,'lMdbyMs':0.8,'lRdo':0.5,'lRdi':0.5,'Zdisc':0.5,'Bdisc':0.5,'lalphad':0.5,'lroCav':0.5,'lroAmp':0.5,'Inc':1,'Av':2.5,'distance':0.9} 

#    step = pm.NUTS(state=start) # Instantiate MCMC sampling algorithm
    step = pm.Metropolis([lTs,lMs,ldMenvbyMs,lRenv,ThetaCav,lMdbyMs,lRdo,lRdi,Zdisc,Bdisc,lalphad,lroCav,lroAmp,Inc,Av,distance])
#    step1 = pm.Slice([x,w,z])
#    step2 = pm.Metropolis([z])    
    trace = pm.sample(1000, step, start=start, progressbar=True) # draw 1000 posterior samples using Sampling
#    trace = pm.sample(10000, [step1,step2], start=start, progressbar=True) # draw 1000 posterior samples using Sampling

print('The trace plot')
fig = pm.traceplot(trace, model.vars[:-1]);
fig.show()
raw_input('enter to close..')

# fig = pm.traceplot(trace, lines={'x': 16, 'w': 12, 'z':3.6})
# fig.show()
# print TESTFindFromGrid(start['x'],start['w'],start['z'])
# print ydata
# print TESTFindFromGrid(np.mean(trace['x']),np.mean(trace['w']),np.mean(trace['z']))


