
# coding: utf-8

# In[5]:


import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import image_to_string


# In[11]:


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = Image.open('vrkIj.png')
text = image_to_string(img)
print(text)


# In[ ]:


cap = cv2.VideoCapture('task.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[12]:


import matplotlib.pyplot as plt


# In[14]:


plt.plot([1,2,3], [4,5,6])
plt.show()

