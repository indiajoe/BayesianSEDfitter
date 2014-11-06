####!/usr/bin/env python
""" This model is to fit the SED a Heirachihal Bayesian Model of multi epoch data using Mone Carlo"""
import numpy as np
import pymc as pm
import scipy.interpolate
from DataModel import DataModel

FluxBands = ['BV','BR','BI','2MASSJ','2MASSH','2MASSK']
Appertures = [23,23,23,23,23,23]

SEDModel = DataModel(FluxBands,Appertures)  #Load the SED model

# Load observed flux and corresponding errors
ObservedData = np.genfromtxt('ObservedData.csv',delimiter=',',filling_values=None)
#ObservedData = np.ma.masked_equal(ObservedData, value=None)
ObservedDataSigma = np.genfromtxt('ObservedDataSigma.csv',delimiter=',',filling_values=None)
ObservedDataTau = 1.0/ObservedDataSigma**2
#ObservedDataSigma = np.ma.masked_equal(ObservedDataSigma, value=None)
T_nu = 3.0  # nu of the T distribution used to model observation. Should be > 2
T_lam = ObservedDataSigma *np.sqrt((T_nu -2.0)/T_nu)

print('Observed: ',ObservedData)
print('ObsError: ',ObservedDataSigma)



def ConvertRealToScaled(param,value):
    """ Converts the real value to scaled value which is used in montecarlo model """
    if SEDModel.MinMaxLimitsLog[param]['log']:
        value = np.log10(value)
    return (value-SEDModel.MinMaxLimitsLog[param]['min'])/(SEDModel.MinMaxLimitsLog[param]['max']-SEDModel.MinMaxLimitsLog[param]['min'])
    
def ConvertScaledToReal(param,value):
    """ Converts the real value to scaled value which is used in montecarlo model """
    value = value*(SEDModel.MinMaxLimitsLog[param]['max']-SEDModel.MinMaxLimitsLog[param]['min']) + SEDModel.MinMaxLimitsLog[param]['min']
    if SEDModel.MinMaxLimitsLog[param]['log']:
        value = 10**value
    return value
    


# Compile and keep the Theano functions for each parameter in the following dictionary
GetUpperLowerBoundsFdic = dict()

    
#Priors definitions start here.....
lTs = pm.Uniform('lTs',lower=0,upper =1)  # Log(time_star)  lower=3,upper =7
lMs = pm.Uniform('lMs',lower=0,upper= 1) # Log(M_star)  lower=0.1,upper= 50


def GetUpperLowerBoundsWith_lTs(param):
    """ Returns the functions to calculate the upper and lower bounds based on lTs"""
    def UpperBound(lTs=lTs):
        """ Returns the upper bound """
        return scipy.interpolate.splev(lTs,SEDModel.MaxMinSplrep[param][0])
    def LowerBound(lTs=lTs):
        """ Returns the lower bound """
        return scipy.interpolate.splev(lTs,SEDModel.MaxMinSplrep[param][1])
    return UpperBound,LowerBound

for param in ['ldMenvbyMs','ThetaCav','lMdbyMs','lRdo','lRdi','lroCav'] :
    GetUpperLowerBoundsFdic[param] = GetUpperLowerBoundsWith_lTs(param)

def GetUpperLowerBoundsWith_lMs(param):
    """ Returns the functions to calculate the upper and lower bounds based on lMs"""
    def UpperBound(lMs=lMs):
        """ Returns the upper bound """
        return scipy.interpolate.splev(lMs,SEDModel.MaxMinSplrep[param][0])
    def LowerBound(lMs=lMs):
        """ Returns the lower bound """
        return scipy.interpolate.splev(lMs,SEDModel.MaxMinSplrep[param][1])
    return UpperBound,LowerBound

for param in ['lroAmp'] :
    GetUpperLowerBoundsFdic[param] = GetUpperLowerBoundsWith_lMs(param)


upperMenv = pm.Deterministic(eval=GetUpperLowerBoundsFdic['ldMenvbyMs'][0],name='upperMenv',parents={'lTs':lTs},doc='up',dtype=float) 
lowerMenv = pm.Deterministic(eval=GetUpperLowerBoundsFdic['ldMenvbyMs'][1],name='lowerMenv',parents={'lTs':lTs},doc='down',dtype=float)
ldMenvbyMs = pm.Uniform('ldMenvbyMs',upper= upperMenv, lower= lowerMenv)  # Log( d(M_env)/dt /M_star)

## Renv Upper and lower bound variables
@pm.deterministic(plot=False)
def upperRenv(age=lTs,mass=lMs):
    """ Returns the upper and lower bounds of log(Renv) """
    StarRad, StarTemp = SEDModel.GetRadTemp(age,mass)
    Rnot = (StarTemp/30.0)**2.5 * StarRad /2
    ubound = max(min(10**5,4*Rnot),10**3)
    #Convert to our scaled numbers between 1 and 0
    return ConvertRealToScaled('lRenv',ubound)

@pm.deterministic(plot=False)
def lowerRenv(age=lTs,mass=lMs):
    """ Returns the upper and lower bounds of log(Renv) """
    StarRad, StarTemp = SEDModel.GetRadTemp(age,mass)
    Rnot = (StarTemp/30.0)**2.5 * StarRad /2
    lbound = max(0.9*10**3,Rnot/4.0)  # 0.9 factor to take into considereaciton of no envelope models
    #Convert to our scaled numbers between 1 and 0
    return ConvertRealToScaled('lRenv',lbound)

#upperRenv,lowerRenv = GetRenvUpperLowerBounds(lTs,lMs)
lRenv = pm.Uniform('lRenv',upper= upperRenv,lower= lowerRenv)  # Log(R_env)


upperThetaCav = pm.Deterministic(eval=GetUpperLowerBoundsFdic['ThetaCav'][0],name='upperThetaCav',parents={'lTs':lTs},doc='up') 
lowerThetaCav = pm.Deterministic(eval=GetUpperLowerBoundsFdic['ThetaCav'][1],name='lowerThetaCav',parents={'lTs':lTs},doc='down')
ThetaCav = pm.Uniform('ThetaCav',upper= upperThetaCav, lower= lowerThetaCav)  # Theta of Cavity


upperlMdbyMs = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lMdbyMs'][0],name='upperlMdbyMs',parents={'lTs':lTs},doc='up') 
lowerlMdbyMs = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lMdbyMs'][1],name='lowerlMdbyMs',parents={'lTs':lTs},doc='down') 
lMdbyMs = pm.Uniform('lMdbyMs',upper= upperlMdbyMs, lower= lowerlMdbyMs)  # Log( M_disc /M_star)

upperlRdo = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lRdo'][0],name='upperlRdo',parents={'lTs':lTs},doc='up') 
lowerlRdo = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lRdo'][1],name='lowerlRdo',parents={'lTs':lTs},doc='down') 
lRdo = pm.Uniform('lRdo',upper= upperlRdo, lower= lowerlRdo)  # Log(R_disc_out)

# Upper limit set to be the disc's outer radius.
@pm.deterministic(plot=False)
def upperlRdi(val=lRdo):
    """ Returns the upper limit as the Disc outer radius """
    RealValue = ConvertScaledToReal('lRdo',val)
    return ConvertRealToScaled('lRdi',RealValue)

lowerlRdi = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lRdi'][1],name='lowerlRdi',parents={'lTs':lTs},doc='down')
lRdi = pm.Uniform('lRdi',upper= upperlRdi, lower= lowerlRdi)  # Log(R_disc_in)

def GetUpperLowerBoundsWith_lRdo(param):
    """ Returns the functions to calculate the upper and lower bounds based on lRdo"""
    def UpperBound(lRdo=lRdo):
        """ Returns the upper bound """
        return scipy.interpolate.splev(lRdo,SEDModel.MaxMinSplrep[param][0])
    def LowerBound(lRdo=lRdo):
        """ Returns the lower bound """
        return scipy.interpolate.splev(lRdo,SEDModel.MaxMinSplrep[param][1])
    return UpperBound,LowerBound

for param in ['Zdisc','Bdisc']:
    GetUpperLowerBoundsFdic[param] = GetUpperLowerBoundsWith_lRdo(param)

# Upperlimit always 1
lowerZdisc = pm.Deterministic(eval=GetUpperLowerBoundsFdic['Zdisc'][1],name='lowerZdisc',parents={'lRdo':lRdo},doc='down')
Zdisc = pm.Uniform('Zdisc',upper= 1, lower= lowerZdisc)  # Z_disc

upperBdisc = pm.Deterministic(eval=GetUpperLowerBoundsFdic['Bdisc'][0],name='upperBdisc',parents={'lRdo':lRdo},doc='up') 
# Lowerlimit always 0
Bdisc = pm.Uniform('Bdisc',upper= upperBdisc, lower= 0)  # Beta_disc

lalphad = pm.Uniform('lalphad',lower=0,upper= 1) # Log(alpha_disc)  lower=0.001,upper= 0.1

upperlroAmp = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lroAmp'][0],name='upperlroAmp',parents={'lMs':lMs},doc='up') 
lowerlroAmp = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lroAmp'][1],name='lowerlroAmp',parents={'lMs':lMs},doc='down')
lroAmp = pm.Uniform('lroAmp',upper= upperlroAmp, lower= lowerlroAmp)  # Log( Ambient density)

upperlroCav = pm.Deterministic(eval=GetUpperLowerBoundsFdic['lroCav'][0],name='upperlroCav',parents={'lTs':lTs},doc='down')
# Lower limit should be Ambient density
@pm.deterministic(plot=False)
def lowerlroCav(val=lroAmp,uplimit=upperlroCav):
    """ Returns the lower limit as the Ambient density """
    RealValue = ConvertScaledToReal('lroAmp',val)
    ScaledValue = ConvertRealToScaled('lroCav',RealValue)
    if ScaledValue > uplimit:  #This can happen in case of no envelope situations
        ScaledValue = 0.9 * uplimit  #Setting the lower limit to a value slightly less than upper limit
#        print('lroCav lowlimit:',ScaledValue)  #Debuggg
    return ScaledValue

lroCav = pm.Uniform('lroCav',upper= upperlroCav, lower= lowerlroCav)  # Log(Envelope Cavity density)

Inc = pm.Uniform('Inc',lower=0,upper= 1) # Inclination  lower=18.19,upper= 87.13

distance = pm.Uniform('distance',lower=0.8,upper= 1.1) #Unit is kpc for Robitaille's flux #Change 1.1 to 1 later
Av = pm.Uniform('Av',lower=1,upper= 5) # Interstellar Av is between 1 and 5


### Now we calculate the expected flux from SED Model
@pm.deterministic(plot=False)
def FluxFromTheory(lTs=lTs,lMs=lMs,ldMenvbyMs=ldMenvbyMs,lRenv=lRenv,ThetaCav=ThetaCav,lMdbyMs=lMdbyMs,lRdo=lRdo,lRdi=lRdi,Zdisc=Zdisc,Bdisc=Bdisc,lalphad=lalphad,lroCav=lroCav,lroAmp=lroAmp,Inc=Inc,Av=Av):
    """ Retunrs the expected flux table from Astrophysics model """
    GridParams = np.array((lTs,lMs,ldMenvbyMs,lRenv,ThetaCav,lMdbyMs,lRdo,lRdi,Zdisc,Bdisc,lalphad,lroCav,lroAmp,Inc))
    i, Flux = SEDModel.GetValue(GridParams,Av)
    return Flux


y_hat = FluxFromTheory/distance**2

#Data likelihood
y_like = pm.Normal('y_like',mu=y_hat, tau=ObservedDataTau, value=ObservedData, observed=True)
#    y_like = pm.T('y_like',mu= y_hat, nu=T_nu, lam=T_lam, observed=ObservedData)  # T distribution for robustness

# End of Model definitions here.....

# ModelNodes = [lTs, lMs, upperMenv, lowerMenv, ldMenvbyMs, upperRenv, lowerRenv, lRenv, upperThetaCav, lowerThetaCav, ThetaCav, upperlMdbyMs, lowerlMdbyMs, lMdbyMs, upperlRdo, lowerlRdo, lRdo, upperlRdi, lowerlRdi, lRdi, lowerZdisc, Zdisc, upperBdisc, Bdisc, lalphad, upperlroAmp, lowerlroAmp, lroAmp, upperlroCav, lowerlroCav, lroCav, Inc, distance, Av, FluxFromTheory, y_hat, y_like]
# model = pm.Model(ModelNodes)
# mcmc  = pm.MCMC(model)
# mcmc.sample(1000, 50)


# # Inference...
# #    start = pm.find_MAP() # Find starting value by optimization
# start = {'lTs':0.5,'lMs':0.5,'ldMenvbyMs':0.8,'lRenv':0.5,'ThetaCav':0.5,'lMdbyMs':0.8,'lRdo':0.5,'lRdi':0.5,'Zdisc':0.5,'Bdisc':0.5,'lalphad':0.5,'lroCav':0.5,'lroAmp':0.5,'Inc':1,'Av':2.5,'distance':0.9} 

# #    step = pm.NUTS(state=start) # Instantiate MCMC sampling algorithm
# step = pm.Metropolis([lTs,lMs,ldMenvbyMs,lRenv,ThetaCav,lMdbyMs,lRdo,lRdi,Zdisc,Bdisc,lalphad,lroCav,lroAmp,Inc,Av,distance])
# #    step1 = pm.Slice([x,w,z])
# #    step2 = pm.Metropolis([z])    
# trace = pm.sample(1000, step, start=start, progressbar=True) # draw 1000 posterior samples using Sampling
# #    trace = pm.sample(10000, [step1,step2], start=start, progressbar=True) # draw 1000 posterior samples using Sampling


## End of sampling.....

# print('The trace plot')
# fig = pm.traceplot(trace, mcmc.vars[:-1]);
# fig.show()
# raw_input('enter to close..')
#---------
# fig = pm.traceplot(trace, lines={'x': 16, 'w': 12, 'z':3.6})
# fig.show()
# print TESTFindFromGrid(start['x'],start['w'],start['z'])
# print ydata
# print TESTFindFromGrid(np.mean(trace['x']),np.mean(trace['w']),np.mean(trace['z']))


