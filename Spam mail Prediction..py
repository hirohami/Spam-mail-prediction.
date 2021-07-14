#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraraies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[2]:


#Data Preprocessing

raw_mail_data = pd.read_csv('spam.csv', encoding = 'latin-1')

#Replacing the null value with null string

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')


# In[3]:


#Data shape of dataset.
mail_data.shape


# In[4]:


#Printing some data to see what it actually is
mail_data.head()


# In[5]:


#Labeling spam mail as 0 and Non-spam mail as 1 which is said as ham mail.

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[6]:


#Separating the data as text and label .

#X-->text 
#Y-->label

X = mail_data['Text']
Y = mail_data['Category']


# In[7]:


print(X)


# In[8]:


print(Y)


# Data splition

# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6, test_size=0.4, random_state=3)


# Extracting the feature

# In[10]:


#Transforming the test data to feature vectors which will be used as input
#to the SVM model using TfidfVectorizer

#Converting the text to lower case letters

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#Converting Y_train and Y_test values as integer
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# # Training the Support Vector Machine model

# In[11]:


model = LinearSVC()
model.fit(X_train_features, Y_train)


# Evaluating the model

# In[12]:


#Performing prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[13]:


print('Accuracy on training data is as follows: ', accuracy_on_training_data)


# In[14]:


#Performing prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[15]:


print('Accuracy on testing data is as follows: ', accuracy_on_test_data)


# Predictions on new mail.

# In[16]:


input_mail = ["?? Hurray you won 1lakh rupees"]

#Here we have to convert text to feature vectors

input_mail_features = feature_extraction.transform(input_mail)

#Let's make model prediction

prediction = model.predict(input_mail_features)
print(prediction)

if (prediction[0] == 1):
    print('HAM mail')
else:
    print('SPAM mail')


# In[ ]:




