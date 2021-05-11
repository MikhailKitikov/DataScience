
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier


# In[21]:


train = pd.read_csv("train.tsv", sep='\t', header=None)
test = pd.read_csv("test.tsv", sep='\t', header=None)


# In[22]:


X = train.drop(train.columns[-1], axis=1)
y = train[train.columns[-1]]


# In[2]:


import seaborn as sb
import matplotlib.pyplot as plt


# In[16]:


data = sb.load_dataset("tips")


# In[17]:


data.columns


# In[19]:


plt.style.use('fivethirtyeight')
plt.hist(data['total_bill'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('total_bill'); plt.ylabel('Number of Buildings');
plt.title('Energy Star Score Distribution');
plt.show()


# In[60]:


X = pd.DataFrame(tips[['sex']])
y = pd.DataFrame(tips[['tip']])

X.fillna(0, inplace=True)


# In[59]:


plt.plot(X)


# In[9]:


model = CatBoostClassifier(task_type="CPU")
model.fit(X, y)


# In[26]:


arr = np.array(model.get_feature_importance(type='FeatureImportance'))


# In[32]:


a = list((arr[i], i) for i in range(len(arr)))
a = sorted(a, reverse = True)


# In[33]:


cleanX = X[X.columns[[5, 95]]]


# In[34]:


clean_model = CatBoostClassifier()
clean_model.fit(cleanX, y)


# In[35]:


cleanTest = test[test.columns[[5, 95]]]
ans = clean_model.predict(cleanTest)


# In[36]:


with open("answer.tsv", "w") as writer:
    for x in ans:
        writer.write(str(int(round(x))) + "\n")


# In[37]:


import matplotlib.pyplot as plt
    
fig, ax = plt.subplots(nrows=20, ncols=5)

for i, row in enumerate(ax):
    for j, col in enumerate(row):
        print(X[X.columns[i * 5 + j]], y)
        #col.plot(X[X.columns[i * 5 + j]], y)

#plt.show()


# In[44]:


print(sum(1 if x == y[i] else 0 for i, x in enumerate(X[X.columns[0]])))

