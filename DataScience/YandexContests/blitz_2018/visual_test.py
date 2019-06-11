
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


data = sb.load_dataset("tips")


# In[3]:


data.columns


# In[4]:


plt.style.use('fivethirtyeight')
plt.hist(data['total_bill'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('total_bill'); plt.ylabel('Number of Buildings');
plt.title('Energy Star Score Distribution');
plt.show()


# In[12]:


plt.scatter(data['total_bill'], data[['tip']])
plt.show()

