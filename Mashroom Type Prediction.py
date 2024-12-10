#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
mdt = pd.read_csv(r"C:\datasets\mushroomsDT.csv")
mdt.head()


# In[2]:


mdt.isnull().sum()[mdt.isnull().sum()> 0]


# In[3]:


mdt.info()


# In[4]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[5]:


mdt[mdt.select_dtypes(include= 'object').columns] =mdt[mdt.select_dtypes(include= 'object').columns].apply(le.fit_transform)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


train_mdt, test_mdt = train_test_split( mdt , test_size= .25)


# In[8]:


train_mdt_x = train_mdt.iloc[ : , 1 ::]
train_mdt_y = train_mdt.iloc[ : , 0]

test_mdt_x = test_mdt.iloc[ : , 1 ::]
test_mdt_y = test_mdt.iloc[ : , 0]


# In[9]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[10]:


logreg.fit(train_mdt_x,train_mdt_y )


# In[11]:


pred_test_mdt = logreg.predict(test_mdt_x)
pred_test_mdt


# In[12]:


from sklearn.metrics import confusion_matrix , recall_score , f1_score , precision_score , accuracy_score , classification_report


# In[13]:


confusion_matrix(test_mdt_y,pred_test_mdt )


# In[14]:


print(classification_report(test_mdt_y,pred_test_mdt ))


# In[15]:


recall_score(test_mdt_y,pred_test_mdt )*100





# In[16]:


precision_score(test_mdt_y,pred_test_mdt )*100


# In[17]:


f1_score(test_mdt_y,pred_test_mdt )*100


# In[18]:


accuracy_score(test_mdt_y,pred_test_mdt )*100


# In[19]:


fpr = 44 *100/(44+1005)
fpr


# In[20]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[21]:


dt.fit(train_mdt_x,train_mdt_y)


# In[22]:


pred_test_mdt_dt = dt.predict(test_mdt_x)
pred_test_mdt_dt


# In[23]:


confusion_matrix(test_mdt_y,pred_test_mdt_dt)


# In[24]:


print(classification_report(test_mdt_y,pred_test_mdt))


# In[25]:


dt.feature_importances_


# In[26]:


dt.feature_importances_.sum()


# In[27]:


train_mdt_x.columns


# In[28]:


df_fea_sig = pd.DataFrame()
df_fea_sig['features']= train_mdt_x.columns
df_fea_sig['imp']=dt.feature_importances_
df_fea_sig =df_fea_sig.sort_values(['imp'], ascending= False)
df_fea_sig


# In[29]:


df_fea_sig.imp[0:10].sum()


# In[30]:


df_fea_sig.features[0:10]


# In[31]:


l1 = list(df_fea_sig.features[0:10])
l1


# In[32]:


l1.insert( 0 , 'class')


# In[33]:


l1


# In[34]:


mdt = mdt.loc[ : , l1]


# In[35]:


from sklearn.model_selection import train_test_split

mdt_train, mdt_test = train_test_split( mdt , test_size= .25)


# In[36]:


mdt_train_x = mdt_train.iloc[:, 1::]
mdt_train_y = mdt_train.iloc[:, 0]


mdt_test_x = mdt_test.iloc[:, 1::]
mdt_test_y = mdt_test.iloc[:, 0]


# In[37]:


from sklearn.tree import DecisionTreeClassifier
dt_mdt = DecisionTreeClassifier()


# In[38]:


dt_mdt.fit(mdt_train_x, mdt_train_y)


# In[39]:


pred_test_mdt_dt1 = dt_mdt.predict(mdt_test_x)
pred_test_mdt_dt1


# In[40]:


confusion_matrix(mdt_test_y,pred_test_mdt_dt1)


# In[41]:


recall_score(mdt_test_y,pred_test_mdt_dt1)*100


# In[42]:


print(classification_report(mdt_test_y,pred_test_mdt_dt1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




