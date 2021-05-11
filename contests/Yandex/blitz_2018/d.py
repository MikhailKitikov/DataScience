
# coding: utf-8

# In[31]:


import catboost
from catboost import CatBoostClassifier


# In[18]:


import pandas as pd

df = pd.read_csv("train.txt", sep='\t', header=None)
X = df.drop(df.columns[[0, 1, 2, 3]], axis=1)
y = df[df.columns[1]]


# In[44]:


model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=2)


# In[45]:


model.fit(X, y)


# In[69]:


import numpy as np

arr = np.array(model.get_feature_importance(data=None,
                       type='FeatureImportance',
                       prettified=False,
                       thread_count=-1,
                       verbose=False))


# In[70]:


a = [(arr[i], i) for i in range(len(arr))]
a = sorted(a, reverse = True)


# In[73]:


for x in a[:50]:
    print(x[1])


# In[72]:


a

