#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Import data

# train_dataset = pd.read_csv('train.csv')
# X_test = pd.read_csv('test.csv')
# train_dataset

# ## To split   x train and y train 

# In[9]:


train_dataset.columns # to know column name in  training dataset


# In[12]:


X_train = train_dataset.drop(" Y", axis=1) # drop last coloumn
Y_train= train_dataset[" Y"]


# ## X_train , y_train not in array format

# In[13]:


X_train = np.array(X_train)
X_test =  np.array(X_test)
Y_train = np.array(Y_train)


# ##  Cost function

# In[14]:


def cost(Y,Y_predicted):
    n=len(X_train)
    s=0
    for i in range(n):
        s+=(Y[i]-Y_predicted[i])**2
    return (1/n)*s    


# ## dldm

# In[15]:


def dldm(X,Y,Y_predicted):
    n=len(X_train)
    s=0
    for i in range (n) :
        s+= X[i]*(Y[i]-Y_predicted[i])
    return -(2/n)*s  


# ## dldc

# In[16]:


def dldc(Y,Y_predicted):
    n=len(X_train)
    s=0
    for i in range (n) :
        s+= (Y[i]-Y_predicted[i])
    return -(2/n)*s    


# ## Prediction

# In[17]:


def predicted_Y(m,points,c):
    Y_lst=[]
    
    for i in range(len(points)):
        d = np.dot(m, points[i])
        Y_lst.append(d+c)
    return np.array(Y_lst)    


# ## Optimization 

# In[20]:


learning_rate=0.01
iterations=10000
m=np.random.random(X_train.shape[1])
c=0
for i in range(iterations):
    Y_predicted=predicted_Y(m,X_train,c)
    m=m-learning_rate*dldm(X_train, Y_train,Y_predicted)
    c=c-learning_rate*dldc(Y_train,Y_predicted)


# In[22]:


m,c


# In[19]:


Y_test=predicted_Y(m,X_test,c)
Y_test


# In[ ]:




