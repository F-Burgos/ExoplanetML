#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:00:52 2021

@author: pipe
"""
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import normalize, StandardScaler
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential 
from keras.layers import Dense 


#Importamos los paquetes.
def Preprocessing(df_location):
  print("Working now with: ",df_location)
  df = pd.read_csv(df_location)
  
  X = df.loc[:, (df.columns != 'Unnamed') & (df.columns != 'LABEL') & (df.columns != 'STARID')
          & (df.columns != 'p_el') & (df.columns != 'p_cs')].values
  y = df.loc[:, (df.columns == 'LABEL')].values.flatten()

  un_y, counts = np.unique(y, return_counts = True)
  print(X)
  X_train=np.nan_to_num(X)
 
  
  pl.clf()
  pl.bar(un_y, counts)
  pl.show()
  
  X_train = normalized = normalize(X_train)

  X_train = filtered = ndimage.filters.gaussian_filter(X_train, sigma=10)
  print(X_train)
  #Convertimos todo nan a -99, las tablas en si no deberian tener nan, pero es mejor tenerlo de todas formas.
  

  #Feature scaling
  std_scaler = StandardScaler()
  X_train = scaled = std_scaler.fit_transform(X_train)
  print('AFTER FEATURE SCALING')
  print(X_train)
  #Dimentioanlity reduction
  # pca = PCA() 
  # X_train = pca.fit_transform(X_train)
  # total=sum(pca.explained_variance_)
  # k=0
  # current_variance=0
  # while current_variance/total < 0.9:
  #     current_variance += pca.explained_variance_[k]
  #     k=k+1
  # print(k)
  k=1
  print(X_train.shape)
  #Apply PCA with n_componenets
  # pca = PCA(n_components=k)
  # X_train = pca.fit_transform(X_train)
  # plt.figure()
  # plt.plot(np.cumsum(pca.explained_variance_ratio_))
  # plt.xlabel('Number of Components')
  # plt.ylabel('Variance (%)') #for each component
  # plt.title('Exoplanet Dataset Explained Variance')
  # plt.show()
  
  #Resampling
  print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
  print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))
  sm = SMOTE()
  X_train_res, y_train_res = sm.fit_resample(X_train, y.ravel())
  print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
  print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
  return X_train_res, y_train_res,k

def Preprocessing2(df_location,N_components):
  print("Working now with: ",df_location)
  df = pd.read_csv(df_location)
  
  X = df.loc[:, (df.columns != 'Unnamed') & (df.columns != 'LABEL') & (df.columns != 'STARID')
          & (df.columns != 'p_el') & (df.columns != 'p_cs')].values
  y = df.loc[:, (df.columns == 'LABEL')].values.flatten()

  un_y, counts = np.unique(y, return_counts = True)

  X_train=np.nan_to_num(X)
 
  
  # pl.clf()
  # pl.bar(un_y, counts)
  # pl.show()
  
  X_train = normalized = normalize(X_train)

  X_train = filtered = ndimage.filters.gaussian_filter(X_train, sigma=10)

  #Convertimos todo nan a -99, las tablas en si no deberian tener nan, pero es mejor tenerlo de todas formas.
  

  #Feature scaling
  std_scaler = StandardScaler()
  X_train = scaled = std_scaler.fit_transform(X_train)
  #Dimentioanlity reduction
  # pca = PCA() 
  # X_train = pca.fit_transform(X_train)
  # total=sum(pca.explained_variance_)

  #Apply PCA with n_componenets
  # pca = PCA(n_components=N_components)
  # X_train = pca.fit_transform(X_train)
  # plt.figure()
  # plt.plot(np.cumsum(pca.explained_variance_ratio_))
  # plt.xlabel('Number of Components')
  # plt.ylabel('Variance (%)') #for each component
  # plt.title('Exoplanet Dataset Explained Variance')
  # plt.show()
  
  #Resampling
  # print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
  # print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))
  # sm = SMOTE()
  # X_train_res, y_train_res = sm.fit_resample(X_train, y.ravel())
  # print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
  # print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
  return X_train, y

def RFClassifier(X_send,Y_send,Interval,Trust):
    X_train, X_test, y_train, y_test = train_test_split (X_send, Y_send, test_size=0.4)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit (X_train, y_train)


    y_pred = clf.predict (X_test)

    #Metrics
    accuracy= accuracy_score (y_test, y_pred)
    precision= precision_score (y_test, y_pred,average = 'macro')
    recall= recall_score (y_test, y_pred, average = 'macro')
    f1= f1_score (y_test, y_pred, average = 'macro')
    print("accuracy: ", accuracy_score (y_test, y_pred))
    print( "precision: ", precision_score (y_test, y_pred,average = 'macro'))
    print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
    print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
    cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
    print (cm)

    # Confusion Matrix
    plt.figure(figsize=(13,10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
    plt.title("Confusion Matrix Random Forest Classifier, I="+Interval+', Trust='+Trust,fontsize=15)
    return (accuracy,precision,recall,f1)

def SVMClassifier(X,Y,Interval,Trust):
  X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size=0.4)
  from sklearn.pipeline import make_pipeline
  from sklearn.svm import SVC
  clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
  clf.fit(X_train,y_train)
  y_pred = clf.predict (X_test)
  #Metrics
  accuracy= accuracy_score (y_test, y_pred)
  precision= precision_score (y_test, y_pred,average = 'macro')
  recall= recall_score (y_test, y_pred, average = 'macro')
  f1= f1_score (y_test, y_pred, average = 'macro')
  print("accuracy: ", accuracy_score (y_test, y_pred))
  print( "precision: ", precision_score (y_test, y_pred,average = 'macro'))
  print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
  print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
  cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
  print (cm)

   # #Confusion Matrix
  plt.figure(figsize=(13,10))
  plt.subplot(221)
  sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
  plt.title("Confusion Matrix Suport Vector Machine, I="+Interval+', Trust='+Trust,fontsize=15)
  return (accuracy,precision,recall,f1)

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
def DeepLearningModel(X,y):
    
    from sklearn.model_selection import cross_val_score
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.models import Sequential # initialize neural network library
    from keras.layers import Dense # build our layers library

    classifier = KerasClassifier(build_fn = build_classifier, epochs = 150)
    accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 5, n_jobs = -1)
    mean = accuracies.mean()
    variance = accuracies.std()
    print("Accuracy mean: "+ str(mean))
    print("Accuracy variance: "+ str(variance))
    return mean,variance


#%%
Intervals= [5,10,15,20,25,30,35,40,45,50,60,70,80,90,95,100]
Trusts=[0,1]

path='/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllPer/LombScargle/Data/'
Results_path= path+'Results/'
os.system('mkdir '+Results_path)

#%%


Data_path= path
RF0f1_scores= []
RF0f1_scores_std= []

RF0precision_scores=[]
RF0precision_scores_std=[]

RF0recall_scores=[]
RF0recall_scores_std=[]

RF0acuracies=[]
RF0acuracies_std=[]


SVM0f1_scores= []
SVM0f1_scores_std= []

SVM0precision_scores=[]
SVM0precision_scores_std=[]

SVM0recall_scores=[]
SVM0recall_scores_std=[]

SVM0acuracies=[]
SVM0acuracies_std=[]


RF1f1_scores= []
RF1f1_scores_std= []

RF1precision_scores=[]
RF1precision_scores_std=[]

RF1recall_scores=[]
RF1recall_scores_std=[]

RF1acuracies=[]
RF1acuracies_std=[]

SVM1f1_scores= []
SVM1f1_scores_std= []

SVM1precision_scores=[]
SVM1precision_scores_std=[]

SVM1recall_scores=[]
SVM1recall_scores_std=[]

SVM1acuracies=[]
SVM1acuracies_std=[]

for i in Intervals:
    for j in Trusts:
        Interval= str(i)
        Trust= str(j)
        X,y,k = Preprocessing(Data_path+"Data_Npeaks="+Interval+".csv")
        
        if j== 0:
            ################RANDOM FOREST
            ACC= []
            PRE=[]
            REC= []
            Fe=[]
            for f in range(0,10):
                accuracy,precision,recall,f1=RFClassifier(X,y,Interval,Trust)
                ACC.append(accuracy)
                PRE.append(precision)
                REC.append(recall)
                Fe.append(f1)
            accuracy= np.mean(ACC)
            accuracy_std= np.std(ACC)
            RF0acuracies.append(accuracy)
            RF0acuracies_std.append(accuracy_std)
            
            precision= np.mean(PRE)
            precision_std= np.std(PRE)
            RF0precision_scores.append(precision)
            RF0precision_scores_std.append(precision_std)
            
            recall= np.mean(REC)
            recall_std= np.std(REC)
            RF0recall_scores.append(recall)
            RF0recall_scores_std.append(recall_std)
           
            f1= np.mean(Fe)
            f1_std= np.std(Fe)
            RF0f1_scores.append(f1)
            RF0f1_scores_std.append(f1_std)
###################################################SVM
            ACC= []
            PRE=[]
            REC= []
            Fe=[]
            for f in range(0,10):
                accuracy,precision,recall,f1=SVMClassifier(X,y,Interval,Trust)
                ACC.append(accuracy)
                PRE.append(precision)
                REC.append(recall)
                Fe.append(f1)
                
            accuracy= np.mean(ACC)
            accuracy_std= np.std(ACC)
            SVM0acuracies.append(accuracy)
            SVM0acuracies_std.append(accuracy_std)
            
            precision= np.mean(PRE)
            precision_std= np.std(PRE)
            SVM0precision_scores.append(precision)
            SVM0precision_scores_std.append(precision_std)
            
            recall= np.mean(REC)
            recall_std= np.std(REC)
            SVM0recall_scores.append(recall)
            SVM0recall_scores_std.append(recall_std)
           
            f1= np.mean(Fe)
            f1_std= np.std(Fe)
            SVM0f1_scores.append(f1)
            SVM0f1_scores_std.append(f1_std)
            
        else:
            ################RANDOM FOREST
            ACC= []
            PRE=[]
            REC= []
            Fe=[]
            for f in range(0,10):
                accuracy,precision,recall,f1=RFClassifier(X,y,Interval,Trust)
                ACC.append(accuracy)
                PRE.append(precision)
                REC.append(recall)
                Fe.append(f1)
            accuracy= np.mean(ACC)
            accuracy_std= np.std(ACC)
            RF1acuracies.append(accuracy)
            RF1acuracies_std.append(accuracy_std)
            
            precision= np.mean(PRE)
            precision_std= np.std(PRE)
            RF1precision_scores.append(precision)
            RF1precision_scores_std.append(precision_std)
            
            recall= np.mean(REC)
            recall_std= np.std(REC)
            RF1recall_scores.append(recall)
            RF1recall_scores_std.append(recall_std)
           
            f1= np.mean(Fe)
            f1_std= np.std(Fe)
            RF1f1_scores.append(f1)
            RF1f1_scores_std.append(f1_std)
###################################################SVM
            ACC= []
            PRE=[]
            REC= []
            Fe=[]
            for f in range(0,10):
                accuracy,precision,recall,f1=SVMClassifier(X,y,Interval,Trust)
                ACC.append(accuracy)
                PRE.append(precision)
                REC.append(recall)
                Fe.append(f1)
                
            accuracy= np.mean(ACC)
            accuracy_std= np.std(ACC)
            SVM1acuracies.append(accuracy)
            SVM1acuracies_std.append(accuracy_std)
            
            precision= np.mean(PRE)
            precision_std= np.std(PRE)
            SVM1precision_scores.append(precision)
            SVM1precision_scores_std.append(precision_std)
            
            recall= np.mean(REC)
            recall_std= np.std(REC)
            SVM1recall_scores.append(recall)
            SVM1recall_scores_std.append(recall_std)

            f1= np.mean(Fe)
            f1_std= np.std(Fe)
            SVM1f1_scores.append(f1)
            SVM1f1_scores_std.append(f1_std)

            
dataRF1=[Intervals,RF1f1_scores,RF1precision_scores,RF1acuracies,RF1recall_scores,RF1f1_scores_std,RF1precision_scores_std,RF1acuracies_std,RF1recall_scores_std]
dataRF1=np.transpose(dataRF1)

dataRF0=[Intervals,RF0f1_scores,RF0precision_scores,RF0acuracies,RF0recall_scores,RF0f1_scores_std,RF0precision_scores_std,RF0acuracies_std,RF0recall_scores_std]
dataRF0=np.transpose(dataRF0)

dataSVM1=[Intervals,SVM1f1_scores,SVM1precision_scores,SVM1acuracies,SVM1recall_scores,SVM1f1_scores_std,SVM1precision_scores_std,SVM1acuracies_std,SVM1recall_scores_std]
dataSVM1=np.transpose(dataSVM1)


dataSVM0=[Intervals,SVM0f1_scores,SVM0precision_scores,SVM0acuracies,SVM0recall_scores,SVM0f1_scores_std,SVM0precision_scores_std,SVM0acuracies_std,SVM0recall_scores_std]
dataSVM0=np.transpose(dataSVM0)


d=['Number of Peaks','F1 score',"Precision","Accuracy","Recall",'F1 score STD',"Precision STD","Accuracy STD","Recall STD"]
RF0df= pd.DataFrame(data=dataRF0, columns=d)
RF0df.to_csv(Results_path+'RF0.csv',index=False)
RF1df= pd.DataFrame(data=dataRF1, columns=d)
RF1df.to_csv(Results_path+'RF1.csv',index=False)

SVM0df= pd.DataFrame(data=dataSVM0, columns=d)
SVM0df.to_csv(Results_path+'SVM0.csv',index=False)

SVM1df= pd.DataFrame(data=dataSVM1, columns=d)
SVM1df.to_csv(Results_path+'SVM1.csv',index=False)
print('FINISH')

# %%
#Clasificando con curvas ruidosas

# TrainData= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllPer/DataLog/'
# T1Intervals= [2500,3000,3500,4000,5000,7000,10000,20000,30000]
# T0Intervals=[2500,3000,3500,4000,5000,7000,10000,20000,30000]
# error=[0.001,0.005,0.01,0.05,0.1,0.5,1,2]
# a=0
# for j in error:
#     a=a+1
#     for i in T1Intervals:
#         Interval= str(i)
#         Trust= str(1)
#         bad_data= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/DataTrust/'+"Data_I-"+Interval+"_T-"+Trust+".csv"
#         X,y,k= Preprocessing(bad_data)
#         X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.4)
#         from sklearn.pipeline import make_pipeline
#         from sklearn.svm import SVC
#         clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#         clf.fit(X_train,y_train)
#         y_pred = clf.predict (X_test)
#         #Metrics
#         accuracy= accuracy_score (y_test, y_pred)
#         precision= precision_score (y_test, y_pred)
#         recall= recall_score (y_test, y_pred, average = 'macro')
#         f1= f1_score (y_test, y_pred, average = 'macro')
        
        
#         print("accuracy: ", accuracy_score (y_test, y_pred))
#         print( "precision: ", precision_score (y_test, y_pred))
#         print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
#         print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
#         cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
#         print (cm)

#         #Confusion Matrix
#         plt.figure(figsize=(13,10))
#         plt.subplot(221)
#         sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
#         plt.title("Confusion Matrix Suport Vector Machine, I="+Interval+', Trust='+Trust+', Sigma='+str(j),fontsize=15)

#         os.system('mkdir /home/pipe/Pictures/Resultados/CurvasRuidosas/SVM_T-1_I-'+str(i)+'/')
#         title='Confusion_Matrix_Suport_Vector_Machine_sigma='+str(a)
#         plt.savefig('/home/pipe/Pictures/Resultados/CurvasRuidosas/SVM_T-1_I-'+str(i)+'/'+title,bbox_inches='tight')
#         plt.show()
        
#         X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.4)
#         from sklearn.ensemble import RandomForestClassifier
#         clf = RandomForestClassifier(n_estimators=1000)
#         clf.fit (X_train, y_train)
#         y_pred = clf.predict (X_test)
#         #Metrics
#         accuracy= accuracy_score (y_test, y_pred)
#         precision= precision_score (y_test, y_pred)
#         recall= recall_score (y_test, y_pred, average = 'macro')
#         f1= f1_score (y_test, y_pred, average = 'macro')
        
        
#         print("accuracy: ", accuracy_score (y_test, y_pred))
#         print( "precision: ", precision_score (y_test, y_pred))
#         print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
#         print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
#         cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
#         print (cm)

#         #Confusion Matrix
#         plt.figure(figsize=(13,10))
#         plt.subplot(221)
#         sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
#         plt.title("Confusion Matrix Suport Vector Machine, I="+Interval+', Trust='+Trust+', Sigma='+str(j),fontsize=15)

#         os.system('mkdir /home/pipe/Pictures/Resultados/CurvasRuidosas/RF_T-1_I-'+str(i)+'/')
#         title='Confusion_Matrix_Random_Forest_sigma='+str(a)
#         plt.savefig('/home/pipe/Pictures/Resultados/CurvasRuidosas/RF_T-1_I-'+str(i)+'/'+title,bbox_inches='tight')
#         plt.show()
# a=0
# for j in error:
#     a=a+1
#     for i in T0Intervals:
#         Interval= str(i)
#         Trust= str(0)
#         bad_data= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/DataTrust/'+"Data_I-"+Interval+"_T-"+Trust+".csv"
#         X,y,k= Preprocessing(bad_data)
#         X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.4)
#         from sklearn.pipeline import make_pipeline
#         from sklearn.svm import SVC
#         clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#         clf.fit(X_train,y_train)
#         y_pred = clf.predict (X_test)
#         #Metrics
#         accuracy= accuracy_score (y_test, y_pred)
#         precision= precision_score (y_test, y_pred)
#         recall= recall_score (y_test, y_pred, average = 'macro')
#         f1= f1_score (y_test, y_pred, average = 'macro')
        
        
#         print("accuracy: ", accuracy_score (y_test, y_pred))
#         print( "precision: ", precision_score (y_test, y_pred))
#         print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
#         print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
#         cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
#         print (cm)

#         #Confusion Matrix
#         plt.figure(figsize=(13,10))
#         plt.subplot(221)
#         sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
#         plt.title("Confusion Matrix Suport Vector Machine, I="+Interval+', Trust='+Trust+', Sigma='+str(j),fontsize=15)

#         os.system('mkdir /home/pipe/Pictures/Resultados/CurvasRuidosas/SVM_T-0_I-'+str(i)+'/')
#         title='Confusion_Matrix_Suport_Vector_Machine_sigma='+str(a)
#         plt.savefig('/home/pipe/Pictures/Resultados/CurvasRuidosas/SVM_T-0_I-'+str(i)+'/'+title,bbox_inches='tight')
#         plt.show()
        
#         X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.4)
#         from sklearn.ensemble import RandomForestClassifier
#         clf = RandomForestClassifier(n_estimators=1000)
#         clf.fit (X_train, y_train)
#         y_pred = clf.predict (X_test)
#         #Metrics
#         accuracy= accuracy_score (y_test, y_pred)
#         precision= precision_score (y_test, y_pred)
#         recall= recall_score (y_test, y_pred, average = 'macro')
#         f1= f1_score (y_test, y_pred, average = 'macro')
        
        
#         print("accuracy: ", accuracy_score (y_test, y_pred))
#         print( "precision: ", precision_score (y_test, y_pred))
#         print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
#         print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
#         cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
#         print (cm)

#         #Confusion Matrix
#         plt.figure(figsize=(13,10))
#         plt.subplot(221)
#         sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
#         plt.title("Confusion Matrix Suport Vector Machine, I="+Interval+', Trust='+Trust+', Sigma='+str(j),fontsize=15)

#         os.system('mkdir /home/pipe/Pictures/Resultados/CurvasRuidosas/RF_T-0_I-'+str(i)+'/')
#         title='Confusion_Matrix_Random_Forest_sigma='+str(a)
#         plt.savefig('/home/pipe/Pictures/Resultados/CurvasRuidosas/RF_T-0_I-'+str(i)+'/'+title,bbox_inches='tight')
#         plt.show()
# %%

#Experimentando con datos empeorados artificialmente. 
#Entrenamos a nuestros mejores clasificadores, los enviamos a clasificar estos datos empeorados
#veamos que tal les va.


TrainData= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllPer/LombScargle/Data/'
# T1Intervals= [5,10,15,20,25,30,35,40,45,50,60,70,80,90,95,100]
T1Intervals=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,95,100]
error=[0.001,0.005,0.01,0.05,0.1,0.5,1,2,3,5,10,100]
a=0
allf1RF=[]
errorlist=[]
allf1SVM=[]
Peakslist=[]
# Training with NoTrust Curves
for i in T1Intervals:
    Interval= str(i)
    Trust=str(1)
    data = TrainData+"Data_Npeaks="+Interval+".csv"
    X,y,k= Preprocessing(data)
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.4)
    print(X_train)
    
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train,y_train)
    y_pred = clf.predict (X_test)
    #Metrics
    accuracy= accuracy_score (y_test, y_pred)
    precision= precision_score (y_test, y_pred)
    recall= recall_score (y_test, y_pred, average = 'macro')
    f1= f1_score (y_test, y_pred, average = 'macro')
    
    

    cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
    #Confusion Matrix
    plt.figure(figsize=(13,10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
    plt.title("Confusion Matrix Suport Vector Machine, Number of Peaks="+Interval+', Trust= None',fontsize=15)

    os.system('mkdir /home/pipe/Pictures/Resultados/CurvasGaussianas/SVM_Npeaks='+str(i)+'/')
    title='Confusion_Matrix_Suport_Vector_Machine_Number_of_peaks='+Interval
    plt.savefig('/home/pipe/Pictures/Resultados/CurvasGaussianas/SVM_Npeaks='+str(i)+'/'+title,bbox_inches='tight')
    plt.show()
    a=1
    F1=[]
    ACC=[]
    PRE=[]
    REC=[]
    
    for j in error:
        bad_data= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/Data/'+"Data_Npeaks="+Interval+".csv"
        # print(bad_data)
        X_trial,y_trial= Preprocessing2(bad_data,k)
        #Esta es una version de preprocesado que no hace oversampling, para no generar objetos sinteticos
        y_pred=clf.predict(X_trial)
        # print(y_pred.shape,y_trial.shape)
        accuracy= accuracy_score (y_trial, y_pred)
        ACC.append(accuracy)
        precision= precision_score (y_trial, y_pred)
        PRE.append(precision)
        recall= recall_score (y_trial, y_pred, average = 'macro')
        REC.append(recall)
        f1= f1_score (y_trial, y_pred, average = 'macro')
        allf1SVM.append(f1)
        F1.append(f1)
        errorlist.append(j)
        Peakslist.append(i)
        # print("accuracy: ", accuracy_score (y_trial, y_pred))
        # print( "precision: ", precision_score (y_trial, y_pred))
        # print ("recall: ", recall_score (y_trial, y_pred, average = 'macro'))
        # print( "f1: ", f1_score (y_trial, y_pred, average = 'macro'))
        cm = confusion_matrix (y_trial, y_pred) # columns: predicted, rows: true
        # print (cm) 
        
        plt.figure(figsize=(13,10))
        plt.subplot(221)
        sns.heatmap(confusion_matrix(y_trial,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
        plt.title("Noisy Curves Sigma="+str(j)+" SVM: Number of Peaks ="+Interval+', Trust= None',fontsize=15)
        title="Noisy_Curves_Number_"+str(a)+"_SVM_Npeaks="+Interval
        plt.savefig('/home/pipe/Pictures/Resultados/CurvasGaussianas/SVM_Npeaks='+str(i)+'/'+title,bbox_inches='tight')
        a=a+1
        plt.show()
    df_names=['Sigma','F1 Score','Accuracy','Precision','Recall']
    df_data=[error,F1,ACC,PRE,REC]
    # print(df_data)
    df_data=np.transpose(df_data)
    Results= pd.DataFrame(data=df_data, columns=df_names)
    Results.to_csv('/home/pipe/Pictures/Resultados/CurvasGaussianas/SVM_Npeaks='+str(i)+'/Metrics.csv',index=False)
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit (X_train, y_train)
    y_pred = clf.predict (X_test)
    #Metrics
    accuracy= accuracy_score (y_test, y_pred)
    precision= precision_score (y_test, y_pred)
    recall= recall_score (y_test, y_pred, average = 'macro')
    f1= f1_score (y_test, y_pred, average = 'macro')
    # print("accuracy: ", accuracy_score (y_test, y_pred))
    # print( "precision: ", precision_score (y_test, y_pred))
    # print ("recall: ", recall_score (y_test, y_pred, average = 'macro'))
    # print( "f1: ", f1_score (y_test, y_pred, average = 'macro'))
    cm = confusion_matrix (y_test, y_pred) # columns: predicted, rows: true
    # print (cm)

    #Confusion Matrix
    plt.figure(figsize=(13,10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
    plt.title("Confusion Matrix Random Forest, Npeaks="+Interval,fontsize=15)

    os.system('mkdir /home/pipe/Pictures/Resultados/CurvasGaussianas/RF_Npeaks='+str(i)+'/')
    title='Confusion_Matrix_Random_Forest_I='+Interval
    plt.savefig('/home/pipe/Pictures/Resultados/CurvasGaussianas/RF_Npeaks='+str(i)+'/'+title,bbox_inches='tight')
    plt.show()
    F1=[]
    ACC=[]
    PRE=[]
    REC=[]
    a=1
    for j in error:
        bad_data= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/Data/'+"Data_Npeaks="+Interval+".csv"
        # print(bad_data)
        X_trial,y_trial= Preprocessing2(bad_data,k)
        #Esta es una version de preprocesado que no hace oversampling, para no generar objetos sinteticos
        y_pred=clf.predict(X_trial)
        # print(y_pred.shape,y_trial.shape)
        accuracy= accuracy_score (y_trial, y_pred)
        ACC.append(accuracy)
        precision= precision_score (y_trial, y_pred)
        PRE.append(precision)
        recall= recall_score (y_trial, y_pred, average = 'macro')
        REC.append(recall)
        f1= f1_score (y_trial, y_pred, average = 'macro')
        allf1RF.append(f1)
        F1.append(f1)
        # print("accuracy: ", accuracy_score (y_trial, y_pred))
        # print( "precision: ", precision_score (y_trial, y_pred))
        # print ("recall: ", recall_score (y_trial, y_pred, average = 'macro'))
        # print( "f1: ", f1_score (y_trial, y_pred, average = 'macro'))
        cm = confusion_matrix (y_trial, y_pred) # columns: predicted, rows: true
        # print (cm) 
        
        plt.figure(figsize=(13,10))
        plt.subplot(221)
        sns.heatmap(confusion_matrix(y_trial,y_pred),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
        plt.title("Noisy Curves Sigma="+str(j)+" RF: Number of Peaks="+Interval,fontsize=15)
        title="Noisy_Curves_Number_"+str(a)+"_RF_Npeaks="+Interval
        plt.savefig('/home/pipe/Pictures/Resultados/CurvasGaussianas/RF_Npeaks='+str(i)+'/'+title,bbox_inches='tight')
        a=a+1
        plt.show()
    df_names=['Sigma','F1 Score','Accuracy','Precision','Recall']
    df_data=[error,F1,ACC,PRE,REC]
    # print(df_data)
    df_data=np.transpose(df_data)
    Results= pd.DataFrame(data=df_data, columns=df_names)
    Results.to_csv('/home/pipe/Pictures/Resultados/CurvasGaussianas/RF_Npeaks='+str(i)+'/Metrics.csv',index=False)
    
#%%
#Mismo que anterior, pero en lugar de generar graficas y guardar, genera 100 intentos
#y luego guarda el averague y la std, por ultimo se genera una cadena de caracteres
#para usar en el siguiente paso

TrainData= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/AllPer/LombScargle/Data/'
# T1Intervals= [5,10,15,20,25,30,35,40,45,50,60,70,80,90,95,100]
T1Intervals=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,95,100]
error=[0.001,0.005,0.01,0.05,0.1,0.5,1,2,3,5,10,100]
a=0
allf1RF=[]
errorlist=[]
allf1SVM=[]
Peakslist=[]
maxrange=30
# Training with NoTrust Curves
for i in T1Intervals:
    Interval= str(i)
    Trust=str(1)
    data = TrainData+"Data_Npeaks="+Interval+".csv"
    X,y,k= Preprocessing(data)
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.4)
    print(X_train)
    
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train,y_train)
    y_pred = clf.predict (X_test)
    #Metrics

    f1= f1_score (y_test, y_pred, average = 'macro')
    for j in error:
        f1sum=[]
        bad_data= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/Data/'+"Data_Npeaks="+Interval+".csv"
        # print(bad_data)
        for k in range(0,maxrange):
            X_trial,y_trial= Preprocessing2(bad_data,k)
            #Esta es una version de preprocesado que no hace oversampling, para no generar objetos sinteticos
            y_pred=clf.predict(X_trial)
            # print(y_pred.shape,y_trial.shape)
        
            f1= f1_score (y_trial, y_pred, average = 'macro')
            f1sum.append(f1)
        # f1heat= str(np.mean(f1sum))+ u"\u00B1" + str(np.std(f1sum))
        f1heat= np.mean(f1sum)
        allf1SVM.append(f1heat)
        errorlist.append(j)
        Peakslist.append(i)

   
        a=a+1
    
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit (X_train, y_train)
    y_pred = clf.predict (X_test)
 
    for j in error:
        f1sum=[]
        bad_data= '/home/pipe/Clases/Tesis/Periodogramas/final_batch/Test_Noise/Gaussian_sigma='+str(j)+'/Data/'+"Data_Npeaks="+Interval+".csv"
        # print(bad_data)
        for k in range(0,maxrange):
            X_trial,y_trial= Preprocessing2(bad_data,k)
            #Esta es una version de preprocesado que no hace oversampling, para no generar objetos sinteticos
            y_pred=clf.predict(X_trial)
            # print(y_pred.shape,y_trial.shape)
        
            f1= f1_score (y_trial, y_pred, average = 'macro')
            f1sum.append(f1)
        # f1heat= str(np.mean(f1sum))+ u"\u00B1" + str(np.std(f1sum))
        f1heat= np.mean(f1sum)
        allf1RF.append(f1heat)

  
   
        a=a+1


#%%[
database=np.transpose([allf1RF,allf1SVM,errorlist,Peakslist])
columnslist= ['RF','SVM','Error','NPeaks']
Errordf=pd.DataFrame(data=allf1RF,columns=["RF"])
Errordf["SVM"]=allf1SVM
Errordf["Error"]=errorlist
Errordf["NPeaks"]=Peakslist
print(Errordf)

ForRF= ((np.asarray(Errordf["RF"])).reshape(16,12))
ForSVM=((np.asarray(Errordf["SVM"])).reshape(16,12))

GraphRF=Errordf.pivot(index='NPeaks',columns='Error',values='RF')
print(GraphRF)


fig,ax=plt.subplots(figsize=(12,7))
plt.title('Performance acording to the amount of noise RF',fontsize=18)
sns.heatmap(GraphRF,annot=ForRF,cmap='RdYlGn',linewidths=0.30,ax=ax)


GraphSVM=Errordf.pivot(index='NPeaks',columns='Error',values='SVM')
print(GraphRF)


fig,ax=plt.subplots(figsize=(12,7))
plt.title('Performance acording to the amount of noise SVM',fontsize=18)
sns.heatmap(GraphSVM,annot=ForSVM,cmap='RdYlGn',linewidths=0.30,ax=ax)

# %%
