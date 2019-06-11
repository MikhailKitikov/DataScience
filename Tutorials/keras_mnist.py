
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf


# In[8]:


from keras import models, layers


# In[22]:


from keras.datasets import mnist 

(rawtrain_x, train_y), (rawtest_x, test_y) = mnist.load_data()


# Fully connected:

# In[23]:


import numpy as np
from keras.utils import to_categorical

train_x = rawtrain_x.reshape((60000, 28 * 28))
train_x = train_x.astype('float32') / 255
train_y = to_categorical(train_y)

test_x = rawtest_x.reshape((10000, 28 * 28))
test_x = test_x.astype('float32') / 255
test_y = to_categorical(test_y)


# In[24]:


dense_model = models.Sequential()
dense_model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
dense_model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
dense_model.add(layers.Dense(10, activation='softmax'))
dense_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


# In[ ]:


dense_model.fit(train_x, train_y, batch_size=128, epochs=5)


# In[26]:


res = dense_model.evaluate(test_x, test_y)


# Convolutive:

# In[26]:


train_x = rawtrain_x.reshape(60000, 28, 28, 1)
test_x = rawtest_x.reshape(10000, 28, 28, 1)


# In[ ]:


conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
conv_model.add(layers.MaxPooling2D((2,2)))
conv_model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
conv_model.add(layers.MaxPooling2D((2,2)))
conv_model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dense(64, activation='relu'))
conv_model.add(layers.Dense(10, activation='softmax'))
conv_model.summary()

conv_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
("")


# In[28]:


conv_model.fit(train_x, train_y, batch_size=128, epochs=5)

