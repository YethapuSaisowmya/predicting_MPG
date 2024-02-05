#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import pandas as pd
import numpy as np
import streamlit as st
from Prediction import predict


# In[3]:


st.title("Predict Mileage Per Gallon")
st.markdown("Model to  predict MPG of a car")

st.header('Car Features')
col1,col2,col3,col4=st.columns(4)
with col1:
    cylinders = st.slider('cylinders',2,8,1)
    displacement=st.slider('displacement',50,500,10)
with col2:
    horsepower=st.slider('horsepower',50,500,10)
    weight=st.slider('weight',1500,6000,250)
with col3:
    acceleration=st.slider('acceleration',8,25,1)
    modelyear=st.slider('modelyear',70,85,1)
with col4:
    origin=st.slider('origin',1,3,1)


# In[4]:


if st.button('Predict of MPG of car'):
    result=predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,origin]]))
    st.text(result[0])


# In[ ]:




