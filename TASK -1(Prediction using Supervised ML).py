#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Spark Foundation

# # Data Science and Business Analytics Intern

# # Name : Sindhuja P

# # Task : Prediction Using Supervised ML

# # Import Modules:

# In[1]:


#Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


# Setting parameters

sns.set_style(style = 'darkgrid')
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = 10,10


# # Importing data and getting basic info:

# In[4]:


df = pd.read_csv('student.csv')
df.head()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.describe()


# # Scaling the scores variables:

# In[11]:


df['Scores'] = df['Scores']/10
df


# # Data Plotting:

# In[13]:


df.plot(x = 'Hours',y='Scores',style = '*',markersize = 10, color = 'red')
plt.show()


# In[14]:


df['Hours'].hist()


# In[15]:


df['Scores'].hist()


# # Data Preparation:

# In[16]:


x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values


# In[17]:


x


# In[18]:


y


# # Training the Algorithm:

# In[19]:


#splitting the data into training and testing data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2, random_state=0)


# In[20]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[21]:


model.fit(x_train, y_train)


# In[22]:


print("Training Completed")


# # visualizing the data model:

# In[23]:


# plotting the regression line

line = model.coef_*x+model.intercept_
plt.scatter(x,y, color ='purple')
plt.plot(x, line , color='green')
plt.xlabel = ("Percentage Scored")
plt.ylabel = ("Hours Studied")
plt.show()


# In[24]:


plt.scatter(x_test,y_test, color ='purple')
plt.plot(x, line , color='green')
plt.xlabel = ("Percentage Scored")
plt.ylabel = ("Hours Studied")
plt.show()


# # Making Predictions:

# In[25]:


print(x_test)
y_pred = model.predict(x_test)


# In[26]:


y_pred


# In[27]:


y_test


# In[28]:


#Comparing Actual Vs Predicted

df = pd.DataFrame({'Actual':[y_test], 'Predicted':[y_pred]})
df


# In[29]:


Hours = 9.25
own_pred = model.predict([[Hours]])


# In[30]:


print("No.of Hours  ={}".format([[Hours]]))
print("Predicted Score ={}".format(own_pred[0]))


# # Evaluating the Model:

# In[32]:


from sklearn import metrics
print("Mean Absolute Error: ",metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




