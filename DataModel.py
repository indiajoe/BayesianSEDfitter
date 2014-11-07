""" This class is Black Box for the Predicted Data's Model.
It does the job of returning the predicted output for an
input set of parameters.

Inside I could do anything.
For example, here it keeps the SED data of Robitaille model on non uniform grid.
When asked for the deterministically predicted flux it computes nearest
neighbour interpolation and returns the flux. """
import numpy as np
import scipy.interpolate
import scipy.spatial
from EnvelopeFitter import FitBoundryPoints
class DataModel():
    def __init__(self,FilterBands,Appertures):
        self.MinMaxLimitsLog = dict()
        self.MaxMinSplrep = dict()
        self.NDdataTree = None
        self.FluxData = None
        self.Load_Robitaille_SED_Models(FilterBands,Appertures)#['2MASSJ'],[23])

    def Load_Robitaille_SED_Models(self,FilterBands,Appertures):
        """ Loads Robitaille's Grid Model for the input python list of FilterBands and corresponding list of Appertures Index
        Apperture Index Numbers are to be given from aperture_list.ascii corresponding to each filter"""

        FilterFiles = {'BV':'BV_y_full.ascii','BR':'BR_y_full.ascii','BI':'BI_y_full.ascii','2MASSJ':'2J_y_full.ascii','2MASSH':'2H_y_full.ascii','2MASSK':'2K_y_full.ascii','IRAC3':'I3_y_full.ascii','MIPS1':'M1_y_full.ascii'}

        # Central wavelength from http://caravan.astro.wisc.edu/protostars/repository.php
        lambdaDic = {'BV':0.55,'BR':0.64,'BI':0.79,'2MASSJ':1.235,'2MASSH':1.662,'2MASSK':2.159,'IRAC3':5.731,'MIPS1':23.68}

        # Load and interpolate the extinction law
        extlaw = np.loadtxt('extinction_law.ascii')
        ExtK_lambda = scipy.interpolate.interp1d(extlaw[:,0],extlaw[:,1],kind='linear')
        Kv_std = 211.4  # cm^2/g
        self.K_lambdaByK_v = np.array([ExtK_lambda(lambdaDic[band])/Kv_std for band in FilterBands])
 
        #Load list of ND parameters first
        ParameterTableColumns = (2,3,6,7,8,10,11,13,14,16,17,18,19,21)
        ParameterTable = np.genfromtxt('yso_parameters.ascii',usecols=ParameterTableColumns)
        self.RawParameterTable=np.copy(ParameterTable)  #Debug
        # Also load the Age Mass, Radius Temp table seperately
        self.AgeMassRadTemp = np.genfromtxt('yso_parameters.ascii',usecols=(2,3,4,5))

        ## Set the No Envelope cases to nearby with envelope value. This is pureley for computation convenience.
        NoEnvIndex = np.where(ParameterTable[:,3]<0)[0]
        WithEnvMask = ParameterTable[:,3] > 0  #To select models only with Envelop
        # dMenv/dt Lower by a factor of 0.9 in real scale
        ParameterTable[NoEnvIndex,2] = 0.9 * np.sqrt(ParameterTable[NoEnvIndex,1]) * 10**(-9)
        # Renv set to 1 AU instead of minimum 1.0394
        ParameterTable[NoEnvIndex,3] = 0.9 * 1000
        # Theta Cavity set to values between 61 and 62 instead of max 59.998
        ParameterTable[NoEnvIndex,4] = np.random.uniform(low=61,high=62,size=len(NoEnvIndex))#60.2
        # Cavity Density set to 0.9 times min 2.8398 * 10**-22 in real scale
        ParameterTable[NoEnvIndex,11] = 0.9 * 2.8398 * 10**(-22)

        ## Set No Disc cases also to nearby disc case value
        NoDiscIndex = np.where(ParameterTable[:,6]<0)[0]
        WithDiskMask = ParameterTable[:,6] > 0  #To select models only with disk later if needed
        # Mdisk Lower by factor of 0.9 in real scale
        ParameterTable[NoDiscIndex,5] = 0.9 * ParameterTable[NoDiscIndex,1] *2.5357* 10**(-10)
        # Rout of Disc lower by 0.9
        ParameterTable[NoDiscIndex,6] = 0.9 * 1.1687
        # Rinner Disc lower by 0.9
        ParameterTable[NoDiscIndex,7] = 0.9 * 0.01
        # Zdisc lower by 0.9
        ParameterTable[NoDiscIndex,8] = 0.9 * 0.5046
        # Beta disk lower by 0.9
        ParameterTable[NoDiscIndex,9] = 0.9 * 1.0
        # Accretion alpha lower by 0.9
        ParameterTable[NoDiscIndex,10] = 0.9 * 0.001

        #Remove models which doesnot have any disk
        print('Removing Disc less models')# and Envelopless models')
        ParameterTable = ParameterTable[WithDiskMask]# & WithEnvMask]
        self.AgeMassRadTemp = self.AgeMassRadTemp[WithDiskMask]# & WithEnvMask]

        print("Parameter Order based on Robitille's header file: {0}".format(str([col+1 for col in ParameterTableColumns])))

        #Convert some of them to log10 base and Scale all the parameters in range 0 to 1.
        ParameterTable[:,0] = np.log10(ParameterTable[:,0])
        self.MinMaxLimitsLog['lTs'] = {'log':True,'min':np.min(ParameterTable[:,0]),'max':np.max(ParameterTable[:,0])}
        ParameterTable[:,1] = np.log10(ParameterTable[:,1])
        self.MinMaxLimitsLog['lMs'] = {'log':True,'min':np.min(ParameterTable[:,1]),'max':np.max(ParameterTable[:,1])}

        ParameterTable[:,2] = np.log10(ParameterTable[:,2]) - ParameterTable[:,1]  #Converting to log(Menv)-log(Ms)
        self.MinMaxLimitsLog['ldMenvbyMs'] = {'log':True,'min':np.min(ParameterTable[:,2]),'max':np.max(ParameterTable[:,2])}
        ParameterTable[:,3] = np.log10(ParameterTable[:,3])
        self.MinMaxLimitsLog['lRenv'] = {'log':True,'min':np.min(ParameterTable[:,3]),'max':np.max(ParameterTable[:,3])}
        self.MinMaxLimitsLog['ThetaCav'] = {'log':False,'min':np.min(ParameterTable[:,4]),'max':np.max(ParameterTable[:,4])}
        ParameterTable[:,5] = np.log10(ParameterTable[:,5]) - ParameterTable[:,1]  #Converting to log(Mdisc)-log(Ms)
        self.MinMaxLimitsLog['lMdbyMs'] = {'log':True,'min':np.min(ParameterTable[:,5]),'max':np.max(ParameterTable[:,5])}
        ParameterTable[:,6] = np.log10(ParameterTable[:,6])
        self.MinMaxLimitsLog['lRdo'] = {'log':True,'min':np.min(ParameterTable[:,6]),'max':np.max(ParameterTable[:,6])}
        ParameterTable[:,7] = np.log10(ParameterTable[:,7])
        self.MinMaxLimitsLog['lRdi'] = {'log':True,'min':np.min(ParameterTable[:,7]),'max':np.max(ParameterTable[:,7])}
        self.MinMaxLimitsLog['Zdisc'] = {'log':False,'min':np.min(ParameterTable[:,8]),'max':np.max(ParameterTable[:,8])}
        self.MinMaxLimitsLog['Bdisc'] = {'log':False,'min':np.min(ParameterTable[:,9]),'max':np.max(ParameterTable[:,9])}
        ParameterTable[:,10] = np.log10(ParameterTable[:,10])
        self.MinMaxLimitsLog['lalphad'] = {'log':True,'min':np.min(ParameterTable[:,10]),'max':np.max(ParameterTable[:,10])}
        ParameterTable[:,11] = np.log10(ParameterTable[:,11])
        self.MinMaxLimitsLog['lroCav'] = {'log':True,'min':np.min(ParameterTable[:,11]),'max':np.max(ParameterTable[:,11])}
        ParameterTable[:,12] = np.log10(ParameterTable[:,12])
        self.MinMaxLimitsLog['lroAmp'] = {'log':True,'min':np.min(ParameterTable[:,12]),'max':np.max(ParameterTable[:,12])}
        self.MinMaxLimitsLog['Inc'] = {'log':False,'min':np.min(ParameterTable[:,13]),'max':np.max(ParameterTable[:,13])}
        
        # Scale the parameters in the range 0 to 1
        self.ParamsList = ['lTs','lMs','ldMenvbyMs','lRenv','ThetaCav','lMdbyMs','lRdo','lRdi','Zdisc','Bdisc','lalphad','lroCav','lroAmp','Inc']
        for i,param in enumerate(self.ParamsList):
            ParameterTable[:,i] = (ParameterTable[:,i]-self.MinMaxLimitsLog[param]['min'])/(self.MinMaxLimitsLog[param]['max']-self.MinMaxLimitsLog[param]['min'])

        self.AgeMassRadTemp[:,[0,1]] = ParameterTable[:,[0,1]] # Updating the AgeMassRadTemp table also for consitancy
 
        print('Finding Min and Max envelopes of sample space.')
        ### Create a dictionary of spline representation of max and min envelopes.
        self.MaxMinSplrep['ldMenvbyMs'] = (FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,2],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,2],boundryfun=np.min))
        self.MaxMinSplrep['ThetaCav'] = (FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,4],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,4],boundryfun=np.min))
        self.MaxMinSplrep['lMdbyMs'] = (FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,5],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,5],boundryfun=np.min)) 
        self.MaxMinSplrep['lRdo'] = (FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,6],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,6],boundryfun=np.min)) 
        self.MaxMinSplrep['lRdi'] = (FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,7],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,7],boundryfun=np.min)) 
        self.MaxMinSplrep['Zdisc'] = (FitBoundryPoints(ParameterTable[:,6],ParameterTable[:,8],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,6],ParameterTable[:,8],boundryfun=np.min))  # X axis is R_disc_out
        self.MaxMinSplrep['Bdisc'] = (FitBoundryPoints(ParameterTable[:,6],ParameterTable[:,9],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,6],ParameterTable[:,9],boundryfun=np.min))   # X axis is R_disc_out        
        self.MaxMinSplrep['lroCav'] = (FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,11],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,0],ParameterTable[:,11],boundryfun=np.min)) 
        self.MaxMinSplrep['lroAmp'] = (FitBoundryPoints(ParameterTable[:,1],ParameterTable[:,12],boundryfun=np.max),FitBoundryPoints(ParameterTable[:,1],ParameterTable[:,12],boundryfun=np.min))  # X axis is M_star

        # Create cKDTree
        self.NDdataTree = scipy.spatial.cKDTree(ParameterTable)
        print('Loading Fluxes for (Filter,Apperture) : {0}'.format(zip(FilterBands,Appertures)))
        Flux=[]
        for Filt,Apper in zip(FilterBands,Appertures):
            Flux.append(np.genfromtxt(FilterFiles[Filt],usecols=(2*Apper))[WithDiskMask])
        self.FluxData = np.vstack(Flux).T
        
        self.ParameterTable = ParameterTable #Later needed for interpolation

        print('Calculating Radius Temperature relationship for Mass and Age in grid')
        self.RadiusTempRelation()
        
        # Code below if for returning Nearest neighbour interpoilator. To choose uncomment the retunr statement above.
        # NDdataDic=dict()
        # for Filt,Apper in zip(FilterBands,Appertures):
        #     Flux = np.genfromtxt(FilterFiles[Filt],usecols=(2*Apper))
        #     NDdataDic[Filt] = scipy.interpolate.NearestNDInterpolator(ParameterTable,Flux,rescale=True)
#            NDdataDic[Filt] = scipy.interpolate.LinearNDInterpolator(ParameterTable,Flux)#,rescale=True)
        #Return the NDinterpolator dictionary
#        return NDdataDic        
    
    def RadiusTempRelation(self):
        """ Calculates spline representaiton to calculate Radius and Temperate from Mass and Age of star """
        # First we should calculate a unique parameter table by using view trick
        RowCombinedView = np.ascontiguousarray(self.AgeMassRadTemp).view(np.dtype((np.void, self.AgeMassRadTemp.dtype.itemsize * self.AgeMassRadTemp.shape[1])))
        _, idx = np.unique(RowCombinedView, return_index=True)  # Index of unique rows
        UniqAgeMassRadTemp = self.AgeMassRadTemp[idx]
        # Conver Radius to AU units from sun radius
        UniqAgeMassRadTemp[:,2] = UniqAgeMassRadTemp[:,2] * 0.0046491
        print('No: of points to fit Rad and Temp:',UniqAgeMassRadTemp.shape) #Debug
        self.LinNDInt_Rad = scipy.interpolate.LinearNDInterpolator(UniqAgeMassRadTemp[:,0:2], UniqAgeMassRadTemp[:,2])
        self.LinNDInt_Temp = scipy.interpolate.LinearNDInterpolator(UniqAgeMassRadTemp[:,0:2], UniqAgeMassRadTemp[:,3])
        # Also create a nearest neighbour interpolation to fall back if sampling outside the boundry of the grid.
        self.NearNDInt_Rad = scipy.interpolate.NearestNDInterpolator(UniqAgeMassRadTemp[:,0:2], UniqAgeMassRadTemp[:,2])
        self.NearNDInt_Temp = scipy.interpolate.NearestNDInterpolator(UniqAgeMassRadTemp[:,0:2], UniqAgeMassRadTemp[:,3])
        # The spline menthod was found to be too slow.
        # tck_Rad = scipy.interpolate.bisplrep(UniqAgeMassRadTemp[:,0], UniqAgeMassRadTemp[:,1], UniqAgeMassRadTemp[:,2])
        # tck_Temp = scipy.interpolate.bisplrep(UniqAgeMassRadTemp[:,0], UniqAgeMassRadTemp[:,1], UniqAgeMassRadTemp[:,3]) 
        # return tck_Rad, tck_Temp 

    def GetValue(self,Parameters,Av):  ###Filter parameter not needed
        """ Return the Value from the Model for the input Filter and  list of Parameters """
#        return self.NDdata[Filter](*Parameters)
        # Calculate the extinction correction
        ExtCorrection = np.power(10,-0.4*Av*self.K_lambdaByK_v)

        # Calculate a linear interpolation of linearly independent k nearest neigbours
        NoOfReqNeigh = 15 # Number of required linearly independent neighbours = No: of Dimensions + 1
        factor = 2  # Multiplicate factor of the required number to query from KDtree
        ObtainedEnough = False
        Indeps = [0,1] # First two will be definitly independent
        candidate = 2
        while len(Indeps) < NoOfReqNeigh :
            d,index=self.NDdataTree.query(Parameters,k=NoOfReqNeigh*factor)
#            print( NoOfReqNeigh*factor,Indeps)
            #print zip(d,i) # DEBUG...
            while candidate < NoOfReqNeigh*factor :
                if np.linalg.matrix_rank(np.hstack([np.ones((len(Indeps)+1,1)),
                                                    self.ParameterTable[list(index[Indeps])+[index[candidate]],:]])) == len(Indeps)+1:
                    Indeps.append(candidate)
                    if len(Indeps) == NoOfReqNeigh: break
                candidate += 1
            factor += 1

        ParMatrix = np.hstack([np.ones((NoOfReqNeigh,1)),self.ParameterTable[index[Indeps],:]])
        Coeffs = np.linalg.lstsq(ParMatrix,self.FluxData[index[Indeps],:])[0]
        return index[Indeps],np.dot(np.concatenate(([1],Parameters)),Coeffs) * ExtCorrection

    def GetRadTemp(self,Age,Mass):
        """ Retruns Stellar modle consistant Radius and temperature for given Mass and Age """
        Radius = self.LinNDInt_Rad(Age, Mass)
        Temp = self.LinNDInt_Temp(Age, Mass)
        # In case the coordinate is outside the grid sample use nearest neighbour values instead of nan
        if np.isnan(Radius) or np.isnan(Temp):  
            Radius = self.NearNDInt_Rad(Age, Mass)
            Temp = self.NearNDInt_Temp(Age, Mass)
            
        # Radius = scipy.interpolate.bisplev(Age, Mass, self.tck_Rad)
        # Temp = scipy.interpolate.bisplev(Age, Mass, self.tck_Temp)
        return Radius, Temp



    def ConvertRealToScaled(self,param,value):
        """ Converts the real value to scaled value which is used in montecarlo model """
        if self.MinMaxLimitsLog[param]['log']:
            value = np.log10(value)
        return (value-self.MinMaxLimitsLog[param]['min'])/(self.MinMaxLimitsLog[param]['max']-self.MinMaxLimitsLog[param]['min'])
    
    def ConvertScaledToReal(self,param,value):
        """ Converts the real value to scaled value which is used in montecarlo model """
        value = value*(self.MinMaxLimitsLog[param]['max']-self.MinMaxLimitsLog[param]['min']) + self.MinMaxLimitsLog[param]['min']
        if self.MinMaxLimitsLog[param]['log']:
            value = 10**value
        return value
    


    def PlotPointsOnGrid(self,PointsArray):
        """ Plots NxD parameter points on grid by taking two parameter pairs each """
        import matplotlib.pyplot as plt

        PlotCombinations=[('lTs','lMs'),('lTs','ldMenvbyMs'),('lTs','lRenv'),('lTs','ThetaCav'),('lTs','lMdbyMs'),('lTs','lRdo'),('lTs','lRdi'),('lRdo','Zdisc'),('lRdo','Bdisc'),('lTs','lalphad'),('lTs','lroCav'),('lTs','lroAmp'),('lTs','Inc')]

        fig, ax = plt.subplots(nrows=4, ncols=4)#, figsize=(6,6))
        for i,(Xpar,Ypar) in enumerate(PlotCombinations):
            XparCol = self.ParamsList.index(Xpar)
            YparCol = self.ParamsList.index(Ypar)
            # First plot all the raw data points with low opacity
            ax[i/4,i%4].plot(self.ParameterTable[:,XparCol],self.ParameterTable[:,YparCol],'.',alpha=0.01,color='k')
            ax[i/4,i%4].set_xlabel(Xpar)
            ax[i/4,i%4].set_ylabel(Ypar)
            #Now plot requested points as points
            ax[i/4,i%4].plot(PointsArray[:,XparCol],PointsArray[:,YparCol],'.',color='r')
        plt.show()
