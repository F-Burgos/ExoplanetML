#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:26:50 2021

@author: pipe
"""

#ARRUINATOR 50000
#USO ULTRA SIMPLE, AGREGA UNA CURVA Y LA ARRUINA 
# VALOR DE LA CURVA ANTIGUO -> VALOR NUEVO +- ELEMENTO DE ERROR

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import os
from astropy.timeseries import LombScargle 

# Predefine pi
pi = np.pi
 
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
#%%

stars= pd.read_csv('noplanetlist.csv')
starname=stars['StarID']
path= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Curves'
targetpath=path+'/planet/'
nontargetpath= path+'/no_planet/'
RV_path= nontargetpath
SNR_list=[]
New_SNR_list= []
Total_Percentage= []
error=[0.001,0.005,0.01,0.05,0.1,0.5,1,2,3,5,10,100]
for j in error:
    output_path= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/'
    os.system('mkdir '+output_path)
    print(RV_path)
    
    for i in starname:
        StarID=i
        Data= np.loadtxt(RV_path+StarID,skiprows=2)
        
        time= Data[:,0]
        vel= Data[:,1]
        errorvel= Data[:,2]

        noise=np.random.normal(0,j,len(vel))
        new_vel= vel+noise
        errorvel=errorvel+ j
        print('Working on ', i)
        spl_word='_'
        StarID=i.partition(spl_word)[0]
        print(StarID)
        ls=LombScargle(time,new_vel,errorvel)
        freq,Valr= ls.autopower()
        fap=ls.false_alarm_probability(power=Valr)

        d= {'Frequency':freq,'Period':1/freq,'LogPeriod':np.log10(1/freq),'Periodogram_Value': Valr,'Periodogram_Value_for_Sorting': Valr,'FalseAlarmProbability':fap}
        df=pd.DataFrame(data=d)
        df.to_csv(output_path+'NP-'+StarID+'.csv',index=False)  
    

    
    
#%%
#Phase 2 GOOO

def SNR(signal,noise):
    snr= np.sqrt(np.mean(signal)**2/((np.mean(noise))**2))
    return snr
SNR_list= []
First_sigma=[]
second_sigma=[]
third_sigma=[]
fourth_sigma=[]
fifth_sigma=[]
sixth_sigma=[]
seventh_sigma=[]
eight_sigma=[]
ninth_sigma=[]
tenth_sigma=[]
eleventh_sigma=[]
twelveth_sigma=[]
ListDic = {
        0.001: First_sigma,
        0.005: second_sigma,
        0.01: third_sigma,
        0.05: fourth_sigma,
        0.1: fifth_sigma,
        0.5: sixth_sigma,
        1: seventh_sigma,
        2: eight_sigma,
        3: ninth_sigma,
        5: tenth_sigma,
        10: eleventh_sigma,
        100: twelveth_sigma,
    }

SNR_list=[]
New_SNR_list= []
Total_Percentage= []
error=[0.001,0.005,0.01,0.05,0.1,0.5,1,2,3,5,10,100]
# error=[0.1]
for i in starname:
    Star= i
    Data= np.loadtxt(RV_path+Star,skiprows=2)
    time= Data[:,0]
    np.nan_to_num(time)
    vel= Data[:,1]
    errorvel= Data[:,2]
    vel=np.nan_to_num(vel)
    errorvel=np.nan_to_num(errorvel)

    SNR=np.sqrt(np.mean(vel)**2/((np.mean(errorvel))**2))
    SNR_list.append(SNR)
    for j in error:
        sigma=j
        print(sigma)
        newvel=vel+np.random.normal(0,j,len(vel))
        SNR_new=np.sqrt(np.mean(newvel)**2)/np.sqrt((sigma+np.mean(errorvel))**2)
        listoappend=ListDic.get(j)
        listoappend.append(SNR_new/SNR)
        
        # print(SNR,SNR_new,SNR_new/SNR *100)
df=pd.DataFrame.from_dict(ListDic)
df['Total SNR']=SNR_list
print(df)
df.to_csv('SNR.csv',index=False)
#%%

print(df['Total SNR'])
error=[0.001,0.005,0.01,0.05,0.1,0.5,1,2,3,5,10,100]

meanSNR=np.mean(df['Total SNR'])
Allsigmasmean=[]
Allsigmasstd=[]
percentualsnr=[]
for i in error:
    Allsigmasmean.append(np.mean(df[i])*meanSNR)
    # Allsigmasmean.append(np.mean(df[i])*100)
    Allsigmasstd.append(np.std(df[i]))
    percentualsnr.append((np.mean(df[i]))/meanSNR *100)
print(meanSNR,df[0.001],Allsigmasmean)
sigma=error

plt.title('Comparative SNR of the modified curves ')
plt.plot(np.linspace(0,110,1000),meanSNR*np.ones(1000),linestyle='dashed',color='r',label='100% of original SNR')
plt.plot(np.linspace(0,110,1000),meanSNR*np.ones(1000)/10,linestyle='dashed',color='b',label='10% of original SNR')
plt.plot(np.linspace(0,110,1000),meanSNR*np.ones(1000)/100,linestyle='dashed',color='g',label='1% of original SNR')
plt.legend()

plt.yscale('log')
plt.ylabel('Signal to Noise Ratio (Logaritmic)')
plt.xscale('log')
plt.xlabel('Value of Sigma (logaritmic)')
print(Allsigmasmean)
plt.scatter(sigma, Allsigmasmean)
plt.grid(True,which='major',axis='both')






