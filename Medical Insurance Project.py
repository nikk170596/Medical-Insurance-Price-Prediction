#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib as plt
#import matplotib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

#pickle
import pickle

#overfitting
from sklearn.linear_model import Ridge, Lasso

#hyperparamter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ### Predict medical insurance price/ premium base on independent features

# In[2]:


df = pd.read_csv("medical_insurance.csv")
df.head()


# In[3]:


#find missing


# In[4]:


df.isna().sum()


# In[5]:


df.isna().mean()


# In[6]:


df.describe()


# ### outlier detection

# In[7]:


df[["age","bmi","children"]].boxplot()


# In[8]:


sns.boxplot(df["bmi"])


# In[9]:


q1 = df["bmi"].quantile(0.25)
q2 = df["bmi"].quantile(0.50)
q3 = df["bmi"].quantile(0.75)
iqr = q3 - q1
uppertail = q3 +1.5 *iqr
print(uppertail)
df.loc[(df["bmi"]>uppertail)]


# In[10]:


df["sex"].unique()


# In[11]:


df["region"].unique()


# In[12]:


df.corr()


# ### vif

# In[13]:


df1 = df.drop(["charges","smoker","sex","region"],axis=1)
x_constant =sm.add_constant(df1)
vif_list = [variance_inflation_factor(x_constant.values ,i) for i  in range(x_constant.shape[1])]
s1 = pd.Series(vif_list,index = x_constant.columns)
s1


# In[14]:


df1 = df.drop(["charges","smoker","sex","region"],axis=1)
x_constant =sm.add_constant(df1)
vif_list =[]
for i in range(x_constant.shape[1]):
    vif = variance_inflation_factor(x_constant.values,i)
    vif_list.append(vif)
s1 = pd.Series(vif_list,index = x_constant.columns)
s1


# In[15]:


s1 =pd.Series(vif_list,index = x_constant.columns)
s1.sort_values().plot(kind="barh")


# ### 4) Feature Engineering

# In[16]:


# treament of missing > .fillna()
# outliers >> either transformation or imputation
# rename


# In[17]:


df.rename(columns={"sex":"gender"},inplace= True)


# In[18]:


df.head()


# In[19]:


df["gender"].replace({"male":0,"female":1},inplace =True)


# In[20]:


df["gender"].value_counts()


# In[21]:


df["smoker"].value_counts()


# In[22]:


df["smoker"].replace({"no":0,"yes":1},inplace = True)


# In[23]:


df["smoker"].value_counts()


# ### Encoding

# In[24]:


df.head()


# In[25]:


df = pd.get_dummies(df,columns=["region"])
df.head()


# ### Data Spliting

# In[26]:


x = df.drop("charges",axis =1)
y = df["charges"]


# In[27]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42)


# ### Model Selection

# In[28]:


lr_model = LinearRegression()
lr_model.fit(x_train,y_train)


# ### Evalution

# ### 8.1 Testing

# In[30]:


y_pred_test = lr_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolute error\n", mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is",r2)

adj_r2 = 1-(((1-r2)*(x_test.shape[0]-1))/(x_test.shape[0]-x_test.shape[1]-1))
print("adjusted r2 score", adj_r2)


# ### 8.2 On Training Data

# In[31]:


y_pred_train  = lr_model.predict(x_train) # y_test
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# In[32]:


ridge_model = Ridge(alpha=1.0) # lambda
ridge_model.fit(x_train,y_train)


# In[33]:


#testing 
y_pred_test = ridge_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolute error\n", mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is",r2)

adj_r2 = 1-(((1-r2)*(x_test.shape[0]-1))/(x_test.shape[0]-x_test.shape[1]-1))
print("adjusted r2 score", adj_r2)


# In[34]:


#training
y_pred_train  = ridge_model.predict(x_train) # y_test
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# In[35]:


lasso_model = Lasso( alpha=1.0)
lasso_model.fit(x_train,y_train)


# In[36]:


#testing lasso
y_pred_test = lasso_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolute error\n", mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is",r2)

adj_r2 = 1-(((1-r2)*(x_test.shape[0]-1))/(x_test.shape[0]-x_test.shape[1]-1))
print("adjusted r2 score", adj_r2)


# In[37]:


#training lasso 
y_pred_train  = lasso_model.predict(x_train) # y_test
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# ### check slope for feature selection

# In[38]:


lasso_model.coef_


# In[39]:


s1 = pd.Series(lasso_model.coef_,index = x.columns)
s1.sort_values().plot(kind = "barh")


# In[ ]:


#gender as 0 scole so we can drop it
 


# In[40]:


x_train1 = x_train.drop("gender",axis=1)


# In[41]:


lasso_model = Lasso(alpha=1.0)
lasso_model.fit(x_train1,y_train)


# In[42]:


#testing lasso
x_test1 =  x_test.drop("gender",axis=1)
y_pred_test = lasso_model.predict(x_test1)
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolute error\n", mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is",r2)

adj_r2 = 1-(((1-r2)*(x_test1.shape[0]-1))/(x_test1.shape[0]-x_test1.shape[1]-1))
print("adjusted r2 score", adj_r2)


# In[43]:


# ridge and lasso no change in MSE or R2


# ### Hyperparameter tuning 

# In[58]:


np.arange(0.01,3,0.01)


# In[44]:


est_ridge = Ridge()
parameter_grid = {"alpha":np.arange(0.01,3,0.01)}
gdsearchCv = GridSearchCV(estimator=est_ridge,param_grid=parameter_grid,cv=5)
gdsearchCv.fit(x_train,y_train)
gdsearchCv.best_estimator_


# In[ ]:





# In[45]:


ridge_model = gdsearchCv.best_estimator_
y_pred_test = ridge_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolute error\n", mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is",r2)

adj_r2 = 1-(((1-r2)*(x_test.shape[0]-1))/(x_test.shape[0]-x_test.shape[1]-1))
print("adjusted r2 score", adj_r2)


# In[46]:


#training
ridge_model = gdsearchCv.best_estimator_
y_pred_train  = ridge_model.predict(x_train) # y_test
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# ### RandomizedSearchCV

# In[50]:


est_ridge = Ridge()
parameter_grid = {"alpha":np.arange(0.01,3,0.01)}
rdsearchCv = RandomizedSearchCV(est_ridge,parameter_grid,cv=5)
rdsearchCv.fit(x_train,y_train)
rdsearchCv.best_estimator_


# In[51]:


# On testing data
ridge_model = rdsearchCv.best_estimator_ # alpha = 0.8200000000000001
y_pred_test  = ridge_model.predict(x_test) # y_test
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolue error\n",mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_test.shape[0]-1))/(x_test.shape[0]-x_test.shape[1]-1))
print("adjusted r2 score",adj_r2)


# In[53]:


# on training data
ridge_model = rdsearchCv.best_estimator_ # 
y_pred_train  = ridge_model.predict(x_train) # y_train
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# In[54]:


# on training data
ridge_model = gdsearchCv.best_estimator_ # 
y_pred_train  = ridge_model.predict(x_train) # y_train
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# ### Lasso

# In[55]:


est_lasso = Lasso()
parameter_grid ={"alpha":np.arange(0.01,3,0.01)}
gdsearchCv = GridSearchCV(estimator=est_lasso,param_grid=parameter_grid, cv=5)
gdsearchCv.fit(x_train,y_train)
gdsearchCv.best_estimator_


# In[56]:


# On testing data
lasso_model = gdsearchCv.best_estimator_ # alpha = 0.8200000000000001
y_pred_test  = lasso_model.predict(x_test) # y_test
mse = mean_squared_error(y_test,y_pred_test)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_test,y_pred_test)
print("mean absolue error\n",mae)

r2 = r2_score(y_test,y_pred_test)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_test.shape[0]-1))/(x_test.shape[0]-x_test.shape[1]-1))
print("adjusted r2 score",adj_r2)


# In[60]:


# on training data
lasso_model = gdsearchCv.best_estimator_ # 
y_pred_train  = lasso_model.predict(x_train) # y_train
mse = mean_squared_error(y_train,y_pred_train)
print("mean squared error\n",mse)

mae = mean_absolute_error(y_train,y_pred_train)
print("mean absolue error\n",mae)

r2 = r2_score(y_train,y_pred_train)
print("r2 score is ",r2)

adj_r2 = 1 - (((1-r2)*(x_train.shape[0]-1))/(x_train.shape[0]-x_train.shape[1]-1))
print("adjusted r2 score",adj_r2)


# ### Final model selected based on MSE and R2 score

# In[59]:


#lr_model
with open("Linear_model.pkl","wb") as file:
    pickle.dump(lr_model,file)


# In[63]:


project_data = {"gender" :{"male":0,"female":1},
               "smoker"  :{"yes":1, "no":0},
               "columns":list(x.columns)}
project_data


# In[64]:


import json


# In[66]:


with open("project_data.json","w") as file:
    json.dump(project_data,file)


# In[ ]:




