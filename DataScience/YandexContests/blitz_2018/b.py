
# coding: utf-8

# In[5]:


X = []
Y = []

with open("b_data.txt", "r") as file:
    for line in file:
        x, y = map(float, line.strip().split())
        X.append(x)
        Y.append(y)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[12]:


import math
features = []

for x in X:
    features.append([math.sin(x) ** 2, math.log(x) ** 2, math.sin(x) * math.log(x), x**2])


# In[13]:


model = LinearRegression()
model.fit(features, Y)

coefs = model.coef_


# In[14]:


a = math.sqrt(coefs[0])
b = math.sqrt(coefs[1])
c = coefs[3]

print(a, b, c)

