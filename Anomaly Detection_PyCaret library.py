#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.utils import version
version()
import pandas as pd
import numpy as np


# In[2]:


#Created dataset
dataset  = [23,45,27,76,56,89,23,210,78,43,76,89,2,54,87,12,90,98,345,76,45,14,76,16,17,9]

#Converting to dataframe and assigning col name as Values
df = pd.DataFrame(dataset, columns = ["Values"])


# In[3]:


#import anomaly detection module
from pycaret.anomaly import *

#intialize the setup
outliers = setup(df)


# In[4]:


#intialize the setup
outliers = setup(df , numeric_features = ["Values"])


# In[5]:


#Creating a model
iso_forest = create_model('iforest')
print(iso_forest)


# In[12]:


#plotting the model
plot_model(iso_forest)


# In[7]:


#Assigning the labels
outlier_results = assign_model(iso_forest)
print(outlier_results)


# In[ ]:




