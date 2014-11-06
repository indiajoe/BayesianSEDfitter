#!/usr/bin/env python
""" This is a simple minimiser tool to fit envelop on top and bottom of data distribution in a scatter plot """
import scipy.optimize
import numpy as np
import scipy.interpolate

def FitEnvelope(Xcoo,Ycoo,xlowp,xhighp,p0,Upper=True):
    """ Input:
            Xcoo : X coordinate of points
            Ycoo : Y coordinate of points    
            xlowp : lowest point in X to fit polynomial
            xhighp : highest point in X to fit polynomial
            p0 : Initial estimate of the polynomial coefficents in the order p0[0] + p0[1] *x^1 + p0[2]*x^2 ...
            Upper : True -> Fit Upper Envelope, False-> Fit Lower Envelope
    """
    DataX=Xcoo[(Xcoo>xlowp) & (Xcoo < xhighp)]
    DataY=Ycoo[(Xcoo>xlowp) & (Xcoo < xhighp)]
    def Cost(Coeff):
        Prediction=np.ones(len(DataY))*Coeff[0] #x^0 term
        for i,beta in enumerate(Coeff[1:]):
            Prediction += np.power(DataX,i+1)*beta

        Diff = (Prediction - DataY) if Upper else (DataY - Prediction)
        PointsOnCorrectSide = (Diff > 0) 
        sumofdist = np.sum(Diff[PointsOnCorrectSide])
        cost =  (sumofdist + np.sum(Diff[~PointsOnCorrectSide]) *-1)/max(0.001,len(Diff[PointsOnCorrectSide]))
        print sumofdist,cost
        return cost
    
    res = scipy.optimize.minimize(Cost, p0,options={'xtol': 1e-8, 'disp': True})
    return res


def FitBoundryPoints(Xcoo,Ycoo,xlowp=None,xhighp=None,Noofpoints=100,boundryfun=np.max):
    """ Returns spline representation of a curve to boundry points in the sample
    Input :
            Xcoo : X coordinate of points
            Ycoo : Y coordinate of points    
            xlowp : lowest point in X to fit curve
            xhighp : highest point in X to fit curve
            Noofpoints : Number of points to choose along the boundry for fitting.
            boundryfun : max -> Fit Upper Envelope, min-> Fit Lower Envelope
         """
    if xlowp is None: xlowp = np.min(Xcoo)
    if xhighp is None: xhighp = np.max(Xcoo)
    XbinBounds=np.linspace(xlowp,xhighp,Noofpoints+1)
    BinlimitY=[]
    BinlimitX=[]
    for i in range(Noofpoints):
        BinMask=(Xcoo>XbinBounds[i]) & (Xcoo <=XbinBounds[i+1])
        BinlimitY.append(boundryfun(Ycoo[BinMask]))
        BinlimitX.append(Xcoo[BinMask][np.where(Ycoo[BinMask]==BinlimitY[-1])][0])
    return scipy.interpolate.splrep(np.array(BinlimitX),np.array(BinlimitY),s=0)
    
