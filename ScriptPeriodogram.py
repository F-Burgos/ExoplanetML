#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 03:19:11 2021

@author: pipe
"""
from __future__ import print_function, division

# Copyright (c) 2016-2017 A. Mortier
# Distributed under the MIT License
###################################
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software
# and associated documentation files (the "Software"),
# to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject
# to the following conditions:
 
# The above copyright notice and this permission
# notice shall be included in all
# copies or substantial portions of the Software.
 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
# WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##################################
 
# This code was the basis for the following works
# Mortier et al. 2015
# http://adsabs.harvard.edu/abs/2015A%26A...573A.101M
# Mortier et al. 2017
# http://adsabs.harvard.edu/abs/2017A%26A...601A.110M
# Please cite these works if you use this.
 
# Import necessary packages
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy.pyTiming import pyPeriod
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
# Predefine pi
pi = np.pi
from astropy.timeseries import LombScargle 

def bgls(t, y, err, plow=0.5, phigh=100, ofac=1, jit=0.0,
         dt = None):
 
    '''
    BGLS: Calculates Bayesian General Lomb-Scargle
                      periodogram, normalised with minimum.
 
    t: times
    y: data
    err: error bars
    plow: lowest period to sample
    phigh: highest period to sample
    ofac: oversampling factor
    jit: white noise to be added to the error bars
    dt: time span to be considered
    '''
 
    # Define time span
    if dt == None:
        dt = np.max(t)-np.min(t)
 
    # Amount of frequencies to sample
    amount = (phigh-plow)*dt*ofac/plow/phigh + 1
 
    # Define frequencies
    f = np.linspace(1./phigh, 1./plow, int(amount))
 
    omegas = 2. * pi * f
 
    # Define weights - optional white noise can be added
    err2 = err * err + jit * jit
    w = 1./err2
    W = sum(w)
 
    # Follow algorithm in Mortier et al. 2015
    bigY = sum(w*y)  
 
    p = []
    constants = []
    exponents = []
 
    for i, omega in enumerate(omegas):
        theta = 0.5 * np.arctan2(sum(w*np.sin(2.*omega*t)),
                                 sum(w*np.cos(2.*omega*t)))
        x = omega*t - theta
        cosx = np.cos(x)
        sinx = np.sin(x)
        wcosx = w*cosx
        wsinx = w*sinx
 
        C = sum(wcosx)
        S = sum(wsinx)
 
        YCh = sum(y*wcosx)
        YSh = sum(y*wsinx)
        CCh = sum(wcosx*cosx)
        SSh = sum(wsinx*sinx)
 
        if (CCh != 0 and SSh != 0):
            K = ((C*C*SSh + S*S*CCh - W*CCh*SSh)/
                 (2.*CCh*SSh))
 
            L = ((bigY*CCh*SSh - C*YCh*SSh - S*YSh*CCh)/
                 (CCh*SSh))
 
            M = ((YCh*YCh*SSh + YSh*YSh*CCh)/
                 (2.*CCh*SSh))
 
            constants.append(1./np.sqrt(CCh*SSh*abs(K)))
 
        elif (CCh == 0):
            K = (S*S - W*SSh)/(2.*SSh)
 
            L = (bigY*SSh - S*YSh)/(SSh)
 
            M = (YSh*YSh)/(2.*SSh)
 
            constants.append(1./np.sqrt(SSh*abs(K)))
 
        elif (SSh == 0):
            K = (C*C - W*CCh)/(2.*CCh)
 
            L = (bigY*CCh - C*YCh)/(CCh)
 
            M = (YCh*YCh)/(2.*CCh)
 
            constants.append(1./np.sqrt(CCh*abs(K)))
 
        if K > 0:
            raise RuntimeError('K is positive.\
                                This should not happen.')
 
        exponents.append(M - L*L/(4.*K))
 
    constants = np.array(constants)
    exponents = np.array(exponents)
 
    logp = (np.log10(constants) +
                        (exponents * np.log10(np.exp(1.))))
 
    # Normalise
    logp = logp - min(logp)
 
    # Return array of frequencies and log of probability
    return f, logp
 
def sbgls(t, y, err, obsstart = 5, plow=0.5, phigh=100,
          ofac=1, jit=0.0, fig='no'):
 
    '''
    SBGLS: Calculates Stacked BGLS periodogram
           Can plot figure
 
    t: times
    y: data
    err: error bars
    obsstart: minimum number of observations to start
    plow: lowest period to sample
    phigh: highest period to sample
    ofac: oversampling factor
    jit: white noise to be added to the error bars
    fig: creating figure
    '''
 
    n = len(t)
 
    # Timespan
    dt = np.max(t)-np.min(t)
 
    # Empty lists to fill for sbgls
    freqs = []
    powers = []
    nrobs = []
 
    # Do BGLS for every set of observations
    # and save results
    for i in range(obsstart,n+1):
 
        freq, power = bgls(t[:i], y[:i], err[:i],
                           plow=plow, phigh=phigh,
                           ofac=ofac, jit=jit, dt = dt)
 
        freqs.extend(freq)
        powers.extend(power)
        nrobs.extend(np.zeros(len(freq))+i)
 
    freqs = np.array(freqs)
    powers = np.array(powers)
    nrobs = np.array(nrobs)
 
    # Make figure
    if fig == 'yes':
 
        plt.scatter(1./freqs, nrobs, c=powers,
                    cmap = plt.get_cmap('Reds'),
                    lw=0, marker = 's')
 
        plt.xlim(min(1./freqs),max(1./freqs))
        plt.ylim(min(nrobs),max(nrobs))
 
        plt.xlabel('Period (days)',fontsize=16)
        plt.ylabel('Nr of observations',fontsize=16)
 
        plt.gca().set_xscale('log')
 
        cbar = plt.colorbar()
        cbar.set_label(r'$\log P$',fontsize=14)
 
    return freqs, nrobs, powers



# %%

stars= pd.read_csv('planetlist.csv')
starname=stars['StarID']
path= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Curves'
targetpath=path+'/planet/'
nontargetpath= path+'/no_planet/'
RV_path= targetpath
output_path= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllPer/LombScargle/'
print(RV_path)

for i in starname:
    StarID=i
    Data= np.loadtxt(RV_path+StarID,skiprows=2)
    
    time= Data[:,0]
    vel= Data[:,1]
    errorvel= Data[:,2]
    print('Working on ', i)
    spl_word='_'
    StarID=i.partition(spl_word)[0]
    print(StarID)
    ls=LombScargle(time,vel,errorvel)
    freq,Valr= ls.autopower()
    fap=ls.false_alarm_probability(power=Valr)

    d= {'Frequency':freq,'Period':1/freq,'LogPeriod':np.log10(1/freq),'Periodogram_Value': Valr,'Periodogram_Value_for_Sorting': Valr,'FalseAlarmProbability':fap}
    df=pd.DataFrame(data=d)
    df.to_csv(output_path+'P-'+StarID+'.csv',index=False)
#%%
# Script para crear multiples graficos de periodogramas.
# allstars= pd.read_csv('plist.csv')
# path= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllPer/'


# output_path='/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllGraph/GLS_FAP_Per/'


# for i in starname:
#     StarID=i
#     Data= np.loadtxt(RV_path+StarID,skiprows=2)

#     time= Data[:,0]
#     vel= Data[:,1]
#     errorvel= Data[:,2]
#     print('Working on ', i)

#     # Compute the GLS periodogram with default options.
#     # Choose Zechmeister-Kuerster normalization explicitly
#     clp = pyPeriod.Gls((time, vel, errorvel), norm="ZK")

#     # Print helpful information to screen
#     clp.info()

#     # Define FAP levels of 10%, 5%, and 1%
#     fapLevels = np.array([0.01, 0.005, 0.001])
#     # Obtain the associated power thresholds
#     plevels = clp.powerLevel(fapLevels)
#     plt.figure()
# # and plot power vs. frequency.
#     plt.xlabel("Frequency")
#     plt.ylabel("Power")
#     plt.plot(clp.freq, clp.power, 'b.-')
#     # Add the FAP levels to the plot
#     for i in range(len(fapLevels)):
#         plt.plot([min(clp.freq), max(clp.freq)], [plevels[i]]*2, '--',
#                  label="FAP = %4.1f%%" % (fapLevels[i]*100))
#     plt.legend()
#     plt.title('GLS Periodogram Star P: '+StarID[:-4])
#     plt.savefig(output_path+StarID[:-4]+'.png')
    
    
 #%%
 

stars= pd.read_csv('planetlist.csv')
starname=stars['StarID']
path= '/home/pipe/Clases/Tesis/Periodogramas/final_batch'


print(RV_path)

# for i in starname:
#     StarID=i
#     Data= np.loadtxt(RV_path+StarID,skiprows=2)
    
#     time= Data[:,0]
#     vel= Data[:,1]
#     errorvel= Data[:,2]
#     print('Working on ', i)
#     spl_word='_'
#     StarID=i.partition(spl_word)[0]
#     print(StarID)
#     x,y=bgls(time,vel,errorvel,plow=0.1,phigh=2000)
#     print(x,y)
#     d= {'Frequency':x,'Period':1/x,'LogPeriod':np.log10(1/x),'Periodogram_Value': y}
#     df=pd.DataFrame(data=d)
#     df.to_csv(output_path+'P-'+StarID+'.csv',index=False)
stars= pd.read_csv('planetlist.csv')
targetpath=path+'/Curves/planet/'
nontargetpath= path+'/Curves/no_planet/'
RV_path= targetpath
starname=stars['StarID']
N_list= []
StarList=[]
for i in starname:
    StarID=i
    Data= np.loadtxt(RV_path+StarID,skiprows=2)
    
    time= Data[:,0]
    vel= Data[:,1]
    errorvel= Data[:,2]
    spl_word='_'
    StarID=i.partition(spl_word)[0]
    N_count=len(vel)
    if N_count < 20:
        print( i)
    N_list.append(N_count)
    StarList.append(StarID)
PlanetNames=StarList
PlanetLengthList=N_list 

    
N_list= []
StarList=[]
RV_path= nontargetpath
stars= pd.read_csv('noplanetlist.csv')
starname=stars['StarID']
for i in starname:
    StarID=i
    Data= np.loadtxt(RV_path+StarID,skiprows=2)
    
    time= Data[:,0]
    vel= Data[:,1]
    errorvel= Data[:,2]
    spl_word='_'
    StarID=i.partition(spl_word)[0]
    N_count=len(vel)
    N_list.append(N_count)
    StarList.append(StarID)
    
    
NoPlanetNames=StarList
NoPlanetLengthList=N_list 


plt.figure
Nbins=20
fig,axs = plt.subplots(2,2,sharex=True)
plt.suptitle('Number of observations for each Star.')
fig.tight_layout(pad=1.5)
axs[0,0].hist(PlanetLengthList,bins=Nbins,cumulative=False,rwidth=0.8)
axs[0,0].set_title('L0 Stars')


axs[0,1].set_title('L0 Stars (Cumulative)')
axs[0,1].hist(PlanetLengthList,bins=Nbins,cumulative=True,density=True,rwidth=0.8)

axs[1,0].set_title('L1 Stars ')
# plt.xlabel('Numero de Observaciones')
# plt.ylabel('Numero de curvas')
axs[1,0].hist(NoPlanetLengthList,bins=Nbins,cumulative=False,rwidth=0.8)

axs[1,1].set_title('L1 Stars (Cumulative)')
axs[1,1].hist(NoPlanetLengthList,bins=Nbins,cumulative=True,density=True,rwidth=0.8)


plt.show()