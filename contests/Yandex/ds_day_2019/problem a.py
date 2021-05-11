
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from PIL import Image, ImageDraw


# In[4]:


rawData = pd.read_csv("out.csv")
rawData.head()


# In[5]:


rawData = rawData.drop(rawData.columns[0], axis = 1)


# In[6]:


import re
def tryParseRGB(text):
    if (re.search(r".\d{1,3}.\d{1,3}.\d{1,3}.", text) != None):
        colors = re.split(r"[^\d]", text)
        colors = colors[1:len(colors)-1]
        b = 0
        w = 0
        for color in colors:
            if (int(color) < 20):
                b += 1
            elif (int(color) > 230):
                w += 1
        return (1 if w > b else 0)
    return -1
            


# In[7]:


def tryParseName(text):
    if (re.search(r"[wW]", text)):
        return 1
    elif (re.search(r"[bB]", text)):
        return 0
    else:
        return -1


# In[8]:


def parseData(df, sz):
    for i in range(sz):
        for j in range(sz):
            res = tryParseRGB(str(df.iloc[i,j]))
            if (res == -1):
                res = tryParseName(str(df.iloc[i,j]))
            df.iloc[i,j] = res
            
            
parseData(rawData, rawData.shape[0])


# In[16]:


resultImage = Image.new("RGB", (3700, 3700))

for i in range(37):
    for j in range(37):
        img = Image.new("RGB", (100, 100))
        color = ((255,255,255) if rawData.iloc[i,j] == 1 else (0,0,0))      
        img.paste(color, [0, 0, 100, 100])
        resultImage.paste(img, (i * 100, j * 100))

resultImage.show()
resultImage.save("res.png")

