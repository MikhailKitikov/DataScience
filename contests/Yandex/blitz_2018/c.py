
# coding: utf-8

# In[56]:


import catboost as cb
from catboost import CatBoostClassifier


# In[4]:


import pandas as pd
import numpy as np


# In[48]:


df = pd.read_table("fr_learn.tsv", sep='\t')


# In[49]:


df.drop(['timestamp'], axis=1, inplace=True)


# In[50]:


X = df.drop(['fresh_click'], axis=1)


# In[51]:


X['query'] = X['query'].astype('category')
X.info()


# In[60]:


y.shape


# In[52]:


y = df['fresh_click']


# In[71]:


testX = pd.read_table("fr_test.tsv", sep='\t')
testX['query'] = testX['query'].astype('category')

valx = X[:100]
valy = y[:100]


# In[88]:


model = CatBoostClassifier(iterations=2,
                           learning_rate=1,
                           depth=2)
model.fit(X[:100000], y[:100000], [0], verbose=True)


# In[91]:


res = model.predict(testX[:100])


# In[95]:



    


# In[98]:


with open("detector.tsv", "w") as writer:
    for i in range(200000):
        writer.write("1\n")

