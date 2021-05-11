
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[9]:


train = pd.read_csv("train.tsv", sep='\t', header=None)
test = pd.read_csv("test.tsv", sep='\t', header=None)


# In[10]:


X = train.drop(train.columns[-1], axis=1)
y = train[train.columns[-1]]


# In[11]:


model = LinearRegression()
model.fit(X, y)


# In[12]:


ans = model.predict(test)


# In[21]:


with open("answer.tsv", "w") as writer:
    for x in ans:
        writer.write(str(x) + "\n")


# In[16]:


from catboost import CatBoostRegressor


# In[27]:


lr = CatBoostRegressor(iterations = 500, learning_rate=0.01)


# In[28]:


lr.fit(X, y)


# In[29]:


res = lr.predict(test)


# In[30]:


with open("answer1.tsv", "w") as writer:
    for x in res:
        writer.write(str(x) + "\n")

