
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[27]:


train_clicks = pd.read_csv('train_clicks.csv')
users = train_clicks.user_id[:100000]
items = train_clicks.picture_id[:100000]


# In[10]:


from matplotlib import pyplot as plt
import numpy as np


# In[28]:


freq = {}

for item in items:
    freq[item] = 0
    
for item in items:
    freq[item] += 1
    
plt.plot(np.arange(0, len(freq)), freq.values())
plt.show()


# In[34]:


type((np.ones_like(users), (users, items)))

