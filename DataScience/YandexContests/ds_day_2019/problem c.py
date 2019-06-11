
# coding: utf-8

# In[3]:


import wave
import pandas as pd
import numpy as np
import librosa


# In[4]:


y, sr = librosa.load('task.wav')


# In[28]:


modY = librosa.effects.split(y, 10, np.max, 1, 1)
modY


# In[56]:


result = ''
lastSeg = 0

for seg in modY:
    a = seg[0]
    b = seg[1]
    if (a - lastSeg > 100):
        result = result + ' '
    if (b - a > 3):
        result = result + '-'
    else:
        result = result + '.'
    lastSeg = b
        
print(result[:200])


# In[21]:


code_dict =  {'.-...': '&', '--..--': ',', '....-': '4', '.....': '5',
     '...---...': 'SOS', '-...': 'B', '-..-': 'X', '.-.': 'R',
     '.--': 'W', '..---': '2', '.-': 'A', '..': 'I', '..-.': 'F',
     '.': 'E', '.-..': 'L', '...': 'S', '..-': 'U', '..--..': '?',
     '.----': '1', '-.-': 'K', '-..': 'D', '-....': '6', '-...-': '=',
     '---': 'O', '.--.': 'P', '.-.-.-': '.', '--': 'M', '-.': 'N',
     '....': 'H', '.----.': "'", '...-': 'V', '--...': '7', '-.-.-.': ';',
     '-....-': '-', '..--.-': '_', '-.--.-': ')', '-.-.--': '!', '--.': 'G',
     '--.-': 'Q', '--..': 'Z', '-..-.': '/', '.-.-.': '+', '-.-.': 'C', '---...': ':',
     '-.--': 'Y', '-': 'T', '.--.-.': '@', '...-..-': '$', '.---': 'J', '-----': '0',
     '----.': '9', '.-..-.': '"', '-.--.': '(', '---..': '8', '...--': '3'
     }


# In[57]:


def decodeMorse(morseCode, resultString):    
    for item in morseCode.split(' '):
        if (item in code_dict):
            resultString = resultString + code_dict.get(item)
        else:
            resultString = resultString + '#'
    return resultString

resultString = ' '
decodeMorse(result, resultString)

