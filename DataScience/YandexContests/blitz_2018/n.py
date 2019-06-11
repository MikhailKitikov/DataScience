
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np

n = 0
ans = []

with open("coins.in", "r") as file:
    
    for i, line in enumerate(file):
        if i == 0:
            n = int(line)
        else:
            k = float(line.split()[0])
            m = float(line.split()[1])
            if k == 0:
                ans.append((0, i - 1))
            else: 
                p = m / k
                ans.append((p, i - 1))
                
#ans = sorted(ans, reverse=True)


# In[34]:


ans


# In[8]:


res = [tp[1] for tp in ans]


# In[15]:


with open("coins.out", "w") as writer:
    for x in res:
        writer.write(str(x) + '\n')

