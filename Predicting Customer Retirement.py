#!/usr/bin/env python
# coding: utf-8

# # Step 0 - Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# # Step 1 - Loading dataset

# In[2]:


training_set = pd.read_csv("Bank_Customer_retirement.csv")


# In[3]:


training_set.head()


# In[4]:


training_set.tail()


# In[5]:


training_set.shape


# # Step 2 - Visualizing the data

# In[6]:


sns.pairplot(data = training_set, hue = 'Retire', vars = ['Age', '401K Savings'])


# In[7]:


sns.countplot(x = 'Retire', data = training_set)


# # Step 3 - Model Training

# In[8]:


training_set.drop('Customer ID', axis = 1, inplace = True)


# In[9]:


training_set


# In[10]:


X = training_set.drop('Retire', axis = 1).values


# In[11]:


X


# In[12]:


y = training_set['Retire'].values


# In[13]:


y


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[16]:


from sklearn.svm import SVC


# In[17]:


classifier = SVC()


# In[18]:


classifier.fit(X_train, y_train)


# # Step 4 - Evaluating the model

# In[19]:


from sklearn.metrics import confusion_matrix, classification_report


# In[20]:


y_pred = classifier.predict(X_test)


# In[21]:


cm = confusion_matrix(y_test, y_pred)


# In[22]:


sns.heatmap(cm, annot = True)


# In[23]:


print(classification_report(y_test, y_pred))


# # Step 5 - Improving the model

# In[24]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[44]:


sns.scatterplot(data = X)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[26]:


classifier = SVC()
classifier.fit(X_train, y_train)


# In[27]:


y_pred = classifier.predict(X_test)


# In[28]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)


# In[29]:


print(classification_report(y_test, y_pred))


# # Step 6 - Using Grid Search to tune parameters

# In[30]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}


# In[31]:


from sklearn.model_selection import GridSearchCV


# In[32]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[33]:


grid.fit(X_train,y_train)


# In[34]:


grid.best_params_


# In[35]:


grid.best_estimator_


# In[36]:


grid_predictions = grid.predict(X_test)


# In[37]:


cm = confusion_matrix(y_test, grid_predictions)


# In[38]:


sns.heatmap(cm, annot=True)


# In[39]:


print("Accuracy = ", (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]) * 100, "%")

