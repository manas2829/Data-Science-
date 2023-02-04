#!/usr/bin/env python
# coding: utf-8

# # Q.3>Write a python code to Implement the KNN classifier and Random forest    on Glass Type Classification dataset using scikit-learn. Also check the accuracy/precision/recall/AUC-ROC of the model.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sklearn.__version__


# In[3]:


Glass=pd.read_csv('D:\\Manas_HP Laptop Backup_13112019\\DATA Science Class Video\\18.Week=17 DS Study Meterial\\glass.csv')


# In[4]:


Glass.head()


# In[5]:


Glass.tail()


# In[6]:


Glass.dtypes


# In[7]:


Glass.info()


# In[8]:


Glass.size


# In[9]:


Glass.shape


# In[10]:


Glass.describe()


# In[11]:


Glass['Type'].value_counts()


# In[12]:


g = Glass.hist(figsize = (15,15))


# In[13]:


plt.figure(figsize=(10,8))
sns.heatmap(Glass.iloc[:,1:10].corr(),annot=True,cmap ='RdYlGn')


# In[14]:


sns.countplot(Glass['Type'])
plt.show()


# In[15]:


#Slipt the dataset into Dependent & Independent
X=Glass.iloc[:,0:9].values
Y=Glass.iloc[:,-1].values


# In[16]:


print(Y)


# In[17]:


print(X)


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y)


# In[19]:


from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)
#scaler=StandardScaler().fit(X_train)
#X_train=Scaler.transform(X_train)
#X_test=Scaler.transform(X_test)


# In[20]:


X_train.shape


# In[21]:


Y_train.shape


# In[22]:


X_test.shape


# In[23]:


Y_test.shape


# # KNN classifier

# In[24]:


test_scores = []
train_scores = []

for i in range(1,30):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,Y_train)
    
    train_scores.append(knn.score(X_train,Y_train))
    test_scores.append(knn.score(X_test,Y_test))


# In[25]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,30),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,30),test_scores,marker='o',label='Test Score')


# In[31]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(14)

knn.fit(X_train,Y_train)
#knn.score(X_test,y_test)


# In[32]:


y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.0)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[33]:


cnf_matrix


# In[34]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))


# In[35]:


from sklearn.metrics import roc_curve,auc
y_pred_prob = knn.predict(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=9) ROC curve')
plt.show()


# In[40]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(X_test,y_pred_prob)


# In[39]:


rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)
accuracy_score(Y_test,y_pred)
asc=accuracy_score(Y_test,y_pred)
print('Randomforest Accurecy=',asc)


# In[42]:


msc=mean_squared_error(Y_test,y_pred)
print("meansqure error =",msc)


# In[46]:


print(classification_report(Y_test, y_pred))
cm = confusion_matrix(Y_test, y_pred)
print('Random Forest confusion matrix\n\n', cm)


# In[47]:


y_pred_prob = rfc.predict(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Rfc')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Rfc ROC curve')
plt.show()


# In[48]:


roc_auc_score(X_test,y_pred_prob)

