
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals
from io import open

# Common imports

import os

import tensorflow as tf

import numpy as np
import pandas as pd
import random as rnd
import math

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import h5py

from sklearn.linear_model import Perceptron

import tensorflow as tf
import os
import time
from datetime import timedelta


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[2]:


d = h5py.File('C:/Users/Umar/Desktop/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')


# In[3]:


xset = d['X'] 
yset = d['Y'] 
zset = d['Z'] 



# In[4]:


len(xset)


# ## All SNR, good/difficult classes

# In[5]:


classes = [22,11,8,19,9,23,10,16,3,4,6]#all Good classes
#FM, GMSK, OQPSK, BPSK,8PSK, AM-SSB-SC, 4ASK, AM-DSB-SC, QPSK, OOK, 16QAM

#classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] #all classes (difficult)
#'32PSK','16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK',
# '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM'

classes.sort()
classes


# In[6]:


def setSNR(N):
    clean_xset = []
    clean_yset = []
    clean_zset = []

    g = 0

    for g in classes:
        clean_xset.extend(xset[(g*106496)+N*4096:(g*106496)+(N+1)*4096])
        clean_yset.extend(yset[(g*106496)+N*4096:(g*106496)+(N+1)*4096])
        clean_zset.extend(zset[(g*106496)+N*4096:(g*106496)+(N+1)*4096])
        g = g+1
    return(clean_xset,clean_yset,clean_zset)    
    
    #plt.plot(clean_zset)


# In[7]:


def y_reshape(yset):
    i = 0
    y_reshaped=[] #each index is an int from 1-24
    while i <len(yset):
        y_reshaped.append(int(np.where(yset[i]==1)[0]))
        i = i +1
    return(y_reshaped)


# In[8]:


def make_complex(xset):
    xcom = []
    xc = []
    a = 0

    while a < len(xset):
        b=0
        while b < 1024:
            xc.append(complex(xset[a][b][0],xset[a][b][1]))
            b = b +1
        xcom.append(xc)
        xc = []
        a = a +1
    return(xcom)


# In[9]:


def find_avg_var(xset):
    i = 0
    real_avg =[]
    imagi_avg = []
    while i < len(xset):
        real_avg.append(np.average(xset[i][:][0]))
        imagi_avg.append(np.average(xset[i][:][1]))
        i = i +1
        
    j = 0
    real_var = [] #this should be the only one that they used in the paper
    imagi_var = []
    while j < len(xset):
        real_var.append(np.var(xset[j][:][0]))
        imagi_var.append(np.var(xset[j][:][1]))
        j = j+1
        
    return(real_avg, imagi_avg, real_var, imagi_var)


# In[10]:


#calculates the HOM

def HOM(p,q,xcom):
    
    k = 0
    k2 = 0
    Mpq = [] 
    XY =[]
    arr = []
    while k < len(x):
        j = 0
        while j < 1024:
            arr.append(xcom[k][j]**(p-q) * (xcom[k][j].conjugate())**(q)) #Appends each 1024 part of the signals, after XX*
            j = j+1
        XY.append(arr)#Each 1024-array is placed into an array
        arr = []
        k = k +1
    
    while k2 < len(XY):
        Mpq.append(np.average(XY[k2])) #Takes the average of each of the 8,000 (2.5 mil) long inputs (which are 1024 points)
        k2 = k2 +1 
    
    return(np.array(Mpq))


# In[12]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix
import itertools


modulation_names = classes
#modulation_names = ['analog','digital']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', aspect = 'auto', cmap=cmap)
    plt.title(title, fontsize = 20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize = 20)
    plt.yticks(tick_marks, classes, fontsize = 20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=15,
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.rcParams["figure.figsize"] = [15,15]
    plt.ylabel('True label', fontsize = 20)
    plt.xlabel('Predicted label', fontsize = 20)
    
#def plot_precision_recall(recall, precision, classes):
 #   plt.plot(recall, classes)
 #   plt.plot(precision, classes)


# In[ ]:





# In[13]:


tree_score = []
svc_score = []
knn_score = []
XG_score = []

tree_recall = []
svc_recall = []
knn_recall =[]
XG_recall = []

tree_precision = []
svc_precision = []
knn_precision = []
XG_precision = []

#26 goes through all the SNR levels (-20,-18...26,28,30)
for i in range(26):
    

    x,y,z = setSNR(i)
    #x = x[::100]
    #y = y[::100]
    #z = z[::100]
    
    y_reshaped = y_reshape(y)
    jo = make_complex(x)
    real_avg, imagi_avg, real_var, imagi_var = find_avg_var(x)
   
    M20, M21, M22, M40, M41, M42, M43, M60, M61, M62, M63 = HOM(2,0,jo),HOM(2,1,jo),HOM(2,2,jo),HOM(4,0,jo),HOM(4,1,jo),HOM(4,2,jo),HOM(4,3,jo),HOM(6,0,jo),HOM(6,1,jo),HOM(6,2,jo),HOM(6,3,jo)
    C20 = M20
    C21 = M21

    C40 = M40 - 3*M20*M20
    C41 = M40 - 3*M20*M21
    C42 = M42 - M20*M20 - 2*M21*M21

    C60 = M60 - 15*M20*M40 + 30*M20*M20*M20
    C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20*M20*M21
    C62 = M62 - 6*M20*M42 - 8*M21*M41 - M22*M40 + 6*M20*M20*M22 + 24*M21*M21*M20
    C63 = M63 - 9*M21*M42 + 12*M21*M21*M21 - 3*M20*M43 - 3*M22*M41 + 18*M20*M21*M22
    
    
    d = {'Real_Avg': real_avg, 'Imagi_Avg': imagi_avg, 'Real_Var': real_var,'Imagi_var': imagi_var, 'M(2,0)r':M20.real,
     'M(2,1)r':M21.real,'M(4,0)r':M40.real,'M(4,1)r':M41.real,'M(4,2)r':M42.real,'M(4,3)r':M43.real,'M(6,0)r':M60.real,
     'M(6,1)r':M61.real,'M(6,2)r':M62.real,'M(6,3)r':M63.real,'C(2,0)r':C20.real,'C(2,1)r':C21.real,'C(4,0)r':C40.real,
     'C(4,1)r':C41.real,'C(4,2)r':C42.real,'C(6,0)r':C60.real,'C(6,1)r':C61.real,'C(6,2)r':C62.real,'C(6,3)r':C63.real,
     'M(2,0)i':M20.imag, 'M(2,1)i':M21.imag,'M(4,0)i':M40.imag,'M(4,1)i':M41.imag,'M(4,2)i':M42.imag,'M(4,3)i':M43.imag,
     'M(6,0)i':M60.imag,'M(6,1)i':M61.imag,'M(6,2)i':M62.imag,'M(6,3)i':M63.imag,'C(2,0)i':C20.imag,'C(2,1)i':C21.imag,
     'C(4,0)i':C40.imag,'C(4,1)i':C41.imag,'C(4,2)i':C42.imag,'C(6,0)i':C60.imag,'C(6,1)i':C61.imag,'C(6,2)i':C62.imag,
     'C(6,3)i':C63.imag,
     #'Kurt_amp_r':kurt_ampl_real,'Kurt_phase_r':kurt_phase_real,'Kurt_freq':kurt_freq,
     #'Kurt_amp_i':kurt_ampl_imagi,'Kurt_phase_i':kurt_phase_imagi,
     'mod':y_reshaped}
    
    df = pd.DataFrame(data=d)

    features = list(df)
    features.remove('mod')

    X = df[features].values
    y = df['mod'].values
    
    
#from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
    
   
    #decision tree
    dtree=DecisionTreeClassifier()
    dtree.fit(X_train,y_train)
    y_pred_tree = dtree.predict(X_test)
    tree_score.append(dtree.score(X_test,y_test))
    tree_recall.append(recall_score(y_test, y_pred_tree, average=None))
    tree_precision.append(precision_score(y_test, y_pred_tree, average=None))
    
    
    #svc
    svc = SVC(gamma='auto')
    svc.fit(X_train,y_train)
    y_pred_svc = svc.predict(X_test)
    svc_score.append(svc.score(X_test,y_test))
    svc_recall.append(recall_score(y_test, y_pred_svc, average=None))
    svc_precision.append(precision_score(y_test, y_pred_svc, average=None))
    
    #knn
    neigh = KNeighborsClassifier(n_neighbors=20) #used to do 5, 20 seemed to perform better
    neigh.fit(X_train, y_train)
    knn_score.append(neigh.score(X_test, y_test))
    y_pred_knn = neigh.predict(X_test)
    knn_recall.append(recall_score(y_test, y_pred_knn, average=None))
    knn_precision.append(precision_score(y_test, y_pred_knn, average=None))
    
    #XGBoost
    XG = xgb.XGBClassifier(max_depth=3, n_estimators=150, learning_rate=0.25).fit(X_train, y_train)
    y_pred_XG = XG.predict(X_test)
    XG_score.append(XG.score(X_test,y_test))
    XG_recall.append(recall_score(y_test, y_pred_XG, average=None))
    XG_precision.append(precision_score(y_test, y_pred_XG, average=None))
    
    
    print(i)


# In[14]:


v = -20
SNR_values = []
while v < 31:
    if v%2 == 0:
        SNR_values.append(v)
    v = v+1

SNR_values


# In[ ]:



plt.plot(SNR_values, tree_score)
plt.ylabel('Accuracy', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('Decision Tree Accuracy across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)


# In[ ]:



plt.plot(SNR_values, svc_score)
plt.ylabel('Accuracy', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('SVC Accuracy across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)


# In[ ]:



plt.plot(SNR_values, knn_score)
plt.ylabel('Accuracy', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('KNN Accuracy across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)


# In[15]:


plt.plot(SNR_values, XG_score)
plt.ylabel('Accuracy', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('XGBoost Accuracy across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)


# In[ ]:


plt.plot(tree_recall)


# In[ ]:


#[3, 4, 6, 8, 9, 10, 11, 16, 19, 22, 23]
plt.plot(SNR_values, tree_recall)
plt.ylabel('Recall', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('Decision Tree Recall across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[ ]:



plt.plot(SNR_values, svc_recall)
plt.ylabel('Recall', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('SVC Recall across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[ ]:


plt.plot(SNR_values, knn_recall)
plt.ylabel('Recall', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('KNN Recall across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[16]:


plt.plot(SNR_values, XG_recall)
plt.ylabel('Recall', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('XGBoost Recall across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[ ]:


plt.plot(SNR_values, tree_precision)
plt.ylabel('Precision', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('Decision Tree Precision across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[ ]:


plt.plot(SNR_values, svc_precision)
plt.ylabel('Precision', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('SVC Precision across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[ ]:


plt.plot(SNR_values, knn_precision)
plt.ylabel('Precision', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('KNN Precision across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[17]:


plt.plot(SNR_values, XG_precision)
plt.ylabel('Precision', fontsize = 20)
plt.xlabel('SNR', fontsize = 20)
plt.title('XGBoost Precision across SNR', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.legend(('FM', 'GMSK', 'OQPSK', 'BPSK','8PSK', 'AM-SSB-SC', '4ASK', 'AM-DSB-SC', 'QPSK', 'OOK', '16QAM'),loc='upper left')


# In[19]:


#Currently creates a confusion matrix with the last SNR group (30)
cnf_matrix_XG=confusion_matrix(y_test, y_pred_XG)

plot_confusion_matrix(cnf_matrix_XG,classes = (['32PSK'
'16APSK',
'32QAM',
'FM',
'GMSK',
'32APSK',
'OQPSK',
'8ASK',
'BPSK',
'8PSK',
'AM-SSB-SC',
'4ASK',
'16PSK',
'64APSK',
'128QAM',
'128APSK',
'AM-DSB-SC',
'AM-SSB-WC',
'64QAM',
'QPSK',
'256QAM',
'AM-DSB-WC',
'OOK',
'16QAM']), title = 'XGBoost Confusion Matrix')


# In[ ]:


xgb.cv

xgb.cv(data=df,nrounds=3,nfold=10,metrics=list("rmse","auc"), max_depth = 300, eta = 1)


# In[36]:


XG_score_l


# In[ ]:


zset[4095], zset[4096]


# In[56]:


#Comparing amongst all 4 classifiers
class_scores = [83.4,77.0329698993166,83.12217643069539,95.65392039230006]
plt.bar(['DTree','SVC','KNN','XGBoost'],class_scores)
plt.ylabel('Accuracy %', fontsize = 15)
plt.xlabel('Classifier', fontsize = 20)
plt.title('Classification Scores on Binary Classifer', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 12)

