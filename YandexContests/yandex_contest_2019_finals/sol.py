
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix

import implicit


# Читаем клики пользователей

# In[3]:


train_clicks = pd.read_csv('train_clicks.csv')


# In[4]:


users = train_clicks.user_id
items = train_clicks.picture_id


# Создаём разреженную матрицу пользователь — объект

# In[5]:


user_item = coo_matrix((np.ones_like(users), (users, items)))


# В качестве модели будем использовать разложение матрицы с помощью метода ALS

# In[7]:


model = implicit.als.AlternatingLeastSquares(factors=10, iterations=30)


# In[8]:


model.fit(user_item.T.tocsr())


# Прочитаем идентификаторы пользователей, для которых нам нужно сделать предсказания

# In[10]:


test_users = pd.read_csv('test_users.csv')


# In[11]:


user_item_csr = user_item.tocsr()


# Для каждого пользователя найдём 100 самых релевантных изображений

# In[12]:


rows = []
for user_id in test_users.user_id.values:
    items = [i[0] for i in model.recommend(user_id, user_item_csr, N=100)]
    rows.append(items)


# Отформатируем идентификаторы как нужно для ответа

# In[13]:


test_users['predictions'] = list(map(lambda x: ' '.join(map(str, x)), rows))


# И запишем предсказания в файл

# In[14]:


test_users.to_csv('predictions.csv', index=False)

