#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[29]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\saisowmya\\Desktop\\auto MPG app"')


# In[30]:


df=pd.read_csv("Auto MPG Reg.csv")


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[32]:


df.head()


# In[33]:


df.info()


# In[34]:


#conbert horse power to numeric
df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')


# In[35]:


df.info()


# In[36]:


df['horsepower'].describe()


# In[37]:


df['horsepower']=df['horsepower'].fillna(df['horsepower'].median())


# In[38]:


y=df['mpg']
X = df.drop(['carname','mpg'],axis=1)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


reg_model = LinearRegression().fit(X,y)


# In[41]:


reg_model.score(X,y)


# In[42]:


reg_predict=reg_model.predict(X)


# In[43]:


from sklearn.metrics import mean_squared_error


# In[44]:


np.sqrt(mean_squared_error(y,reg_predict))


# In[45]:


#For deployement , model needs to be saved as .pkl(pickle) file or .sav(joblib) library


# In[46]:


import joblib


# In[47]:


joblib.dump(reg_model,'reg.sav')


# In[ ]:





# In[ ]:




