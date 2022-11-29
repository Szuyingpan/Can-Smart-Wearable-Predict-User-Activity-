#!/usr/bin/env python
# coding: utf-8

# # Can Smart Wearable Predit User Activity?
# 
# September 11, 2022

# # Content 
# 
# 
# ## 1 Introduction 
# Some disease patterns such as high blood pressure, diabetes and arrhythmia demand for increased health accessibility, remote monitoring and early disease detection. Smart wearables have the potential to offer a minimally-obtrusive telemedicine platform for individuals health services that are easily, time-saving, cost-effective and better quality for the patient and citizen[1]. 
# 
# In the previous coursework is mainly focusing in understanding the correlation of data sets including sleepDay, dailyActivity and weightLogInfo from Fitbit Fitness tracker between April 12,2016 and May 12, 2016. There are two evidences that have proven strongly correlation between BMI and sleep time, as well as Steps amounts and activity intensity. Therefore, the research topic in this notebook is aimed to find how smart wearables can predict user activities. In addition, by adding the other competitive Apple smart wearable data set, I can have a clear overview of predicted performance. In latest wearable devices market share, Apple, the top one brand has occupied 30.5% market share during the first quarter of 2022(1Q22) according to new data from International data Corporation (IDC).   
# 
# ### 1.1 Research topic 
# In this proposal, the three data sets are imported from Fitbit Fitness tracker between April 12,2016 and May 12, 2016 and "Replication Data for: Using machine learning methods to predict physical activity types with Apple Watch and Fitbit data using indirect calorimetry as the criterion.", Fuller, Daniel, 2020, in Harvard Dataverse. 
# 
# As machine learning is increasingly used in the medical health research and has potential to advanced our understanding of how to characterize, predict and treat with disease patterns. In recent study, Behav Ther[2] mentioned the first step to determine if (and which) machine learning method(s) to use is specifying one’s research question. Using Hernán and colleagues’ framework, there are three major data science research tasks: description, prediction, and causal inference (Hernán et al., 2019). Machine learning can be used for each of these three tasks, but traditional statistical methods may prove to be sufficient and more appropriate depending on the specific research question[2].  
# 
# In certain questions, Machine learning may offer new tools to overcome challenges for which traditional statistical methods are not well-suited. In this proposal, I decide to use machine learning libraries Numpy and Scikit-learn to build supervised machine learning models for prediction and binary classification, including Random Forest, SVM and Neural Network in validating each model and evaluating performance. With binary classification, for example lying or not lying, sitting or not sitting, I use XGBoost to test the performance. 
# 
# ### 1.2 Model research 
# Data mining[12] is an important and interesting field in Computer Science and has received a lot of attention from the research community particularly over the past decade. It is important because it specializes in analyzing the data from different perspectives and summarizing it into useful information – information that can be used to increase revenues, cut costs, or both. It is interesting because it applies methods at the intersection of multiple disciplines including artificial intelligence, machine learning, statistics, and database systems (Fayyad et al., 1996).
# 
# From my research of preprocessing data, I realised the high accuracy of overfitting in supervised machine learning is a vital issue. In R Roelofs (2019)[3] pointes that“overfitting” is often used as an umbrella term to describe any unwanted performance drop of a machine learning model. The phenomenon of overfitting is ubiquitous throughout statistics. It is especially problematic in nonparametric problems, where some form of regularization is essential in order to prevent it[4]. With Random Forest and SVM in the concepts of overfitting and definition of imbalanced data, I start to explore.  
# 
# ### 1.3 Execution time
# Each type of activity was separately predicted performance with Random Forest and XGBoost classifier. Therefore, some of the code would take time to tun the results. 
# 
# ## 2 Aims and objectives 
# The aim of this project is to explore how smart wearable can reflect user activity by using machine learning model to predict performance. The steps of this projects are:
#    1. import 3 data sets, the old version "dailyActivity' of Fibit dataset and two new datasets of Apple watch           series 2 and Fitbit charge HR2. 
#    2. data mining with cleaning data, correlating between variables and visualization.
#    3. preparing for training data including convert categorical data into numerical data.
#    4. evaluate the data features and analysis though visualization and statistic method.
#    5. validating model and tune parameters iteratively. 
#    6. evaluate the results. 
# 
# 
# ### 2.1 Methods 
# The new data sets according to the data sources[5], they recruited a convenience sample of 46 participants (26 women) to wear three devices, a GENEActiv, and Apple Watch Series 2, a Fitbit Charge HR2. Participants completed a 65-minute protocol with 40-minutes of total treadmill time and 25-minutes of sitting or lying time. Indirect calorimetry was used to measure energy expenditure. The outcome variable for the study was the activity class; lying, sitting, walking self-paced, 3 METS, 5 METS, and 7 METS (Fuller, 2020). I this in project, I plan to focus on comprising Apple watch and Fitbit, the two popular devices mainly. By training each type of activity with machine learning classification, I learn to know what is the possible method for prediction and different types of classification. 
# 
# Additionally, as the types of activity is categorical data, it needs to covert into numerical data type if I plan to predict each activity. Hence, labeling or one hot encoding are the methods fpr solving the problems. Initially, I use label encoding to order each activity. However, I shortly realized I should apply one-hot encoding as there is no order for six types of activity. Each of variable is independent. 
# 
# In the ML model, I intentionally use random forest, SVM, neural network and XGBoost to understand the label or binary classification. As the data is small, I tried to adjust the parameters of random forest in order to avoid overfitting problem. Initially, the accuracy of random forest in sitting, lying, running 3METs and running 7METs are almost 100% accuracy in Apple watch data set. Although the accuracy is not the only case I should determine, I noticed the counts of binary classification is very differentiated. Secondly, through the AUC-ROC curve, the result of AUC line is almost out of the box. According to Kaur, Pannu and Malhi [6] research, imbalanced data classification is a problem in which proportional class sizes of a dataset differs relatively by a substantial margin. In this, one class is at least depicted with just a few numbers of samples (called the minority class) and rest falls into the other class (called the majority class). They mentioned that [6] the most popular methods used for imbalanced data are neighborhood cleaning rule, safe level smote, cost sensitive algorithm and neural networks. In this project, I applied SMOTE method to solve the imbalanced data for each activity and tried different times of setting parameters in case overfitting situation.
# 
# In the model of SVM classifier, the support vector machine is considered [7]a new learning machine for two-group classification problems. The machine conceptually implements the following idea: input vectors are non-linearly mapped to a very high dimension feature space. Support vector machines (SVMs) are excellent kernel-based tools for binary data classification and regression. [8]Its primal problem can be understood in the following way: Construct two parallel supporting hyperplanes such that, on the one hand, the band between the two parallel hyperplanes separates the two classes (the positive and negative data points) well, and on the other hand, the width of this band is maximized, leading to the introduction of the regularization term. Thus, the structural risk minimization principle is implemented. The final separating hyperplane is selected to be the “middle one” between the above two supporting hyperplanes. In my case, giving the overview of data visualization, most of the features are discrete or highly skewed data sets. SVM is sensitive to above and does not performance well. In the process of data training with SVM, I received 100% accuracy on the sitting, running 3METs, Running 7METs and Self Pace walk. Besides, the rest of the accuracy result are perfectly hight with 99.86% of lying and 99.59 of running 5METs respectively in Apple watch data set. For Fitbit data set, the results of accuracy are very similar to Apple watch, the accuracy of score range from 99.23% to 100%.    
# 
# Finally, based on decision trees, I apply XGBoost preprocessing data. XGBoost [9] is a decision tree ensemble based on gradient boosting designed to be highly scalable. Similarly, to gradient boosting, XGBoost builds an additive expansion of the objective function by minimizing a loss function. To my knowledge, numbers of parameters can change the result effectively. With the XGBoost documentation, I noticed the max_depth, gamma, subsample and colsample_bytree can control overfitting in the case of optimising parameters. In the gamma case, the minimum loss reduction required to make a further split. I apply cross validation and purposely increase the gamma values, lower ratio of subsample in order to avoid overfitting. While the data are imbalanced, I try to balance the positive and negative weights via scale_pos_weight and use the AUC for evaluation.  

# In[1]:


# Import required library. 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from scipy.optimize import curve_fit
from scipy.stats import linregress

import xgboost as xgb
from xgboost import XGBClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import GridSearchCV
from matplotlib import cm
from sklearn.metrics import roc_curve, roc_auc_score, auc


# In[2]:


# Install required package 

#!pip install pycaret


# In[3]:


#pip install -U dataprep


# In[4]:


#pip install plotly


# In[5]:


#! pip install mysql-connector-python


# In[6]:


#!pip3 install ipython-sql


# In[7]:


#!pip3 install mysqlclient


# In[8]:


#!pip install lightgbm


# In[9]:


#!pip3 install KMeans


# In[10]:


#!pip install scrapy


# ### Import Fibit dailyActivity Data

# In[11]:


activity = pd.read_csv('dailyActivity.csv')
activity


# In[12]:


activity.info()


# In[13]:


# Check the activity data to see if there is a missing value
activity.isnull().sum()


# In[14]:


activity.isnull().values.any()


# In[15]:


# convert float to int

activity['VeryActiveMinutes'] = activity['VeryActiveMinutes'].astype(int)
print(activity['VeryActiveMinutes'])


# In[16]:


activity['VeryActiveMinutes'].unique()


# In[17]:


# In VeryActiveMinutes column, NaN still remains. 
# Remove all NaN again in VeryActiveMinutes column. 


# In[18]:


activity['VeryActiveMinutes'].dtype


# In[19]:


np.isnan(activity['VeryActiveMinutes'])


# In[20]:


activity.index[np.where(np.isnan(activity['VeryActiveMinutes'])[0])]


# In[21]:


pd.isnull(activity['VeryActiveMinutes']).sum()


# In[22]:


# Replace NaN, inplace True
activity.dropna(subset=['VeryActiveMinutes'],axis=0,inplace=True)
print(activity['VeryActiveMinutes'])


# ### Preprocessing data on VeryActiveMinutes

# In[23]:


# Check the logest minutes which is 210
activity.loc[activity['VeryActiveMinutes']>100]


# In[24]:


ac_mean=activity['VeryActiveMinutes'].mean()
ac_mean=round(ac_mean)


# ### Imputation replaces 0 with mean.

# In[25]:


activity['VeryActiveMinutes'] = activity['VeryActiveMinutes'].replace(0,ac_mean)
print(activity['VeryActiveMinutes'])


# ### Dealing with missing data "0" by replce with mean

# In[26]:


# 0 has been replaced by mean

activity['VeryActiveMinutes'].mean()


# In[27]:


# double check NaN
activity['VeryActiveMinutes'].isnull().values.any()


# In[28]:


activity['VeryActiveMinutes'].isnull().values.sum()


# ### Set Up 2 bins and cut data into very activity and low activity

# In[29]:


bins = (2, 60, 210)
group_names = ['LowActivity','VeryActivity']
activity['VeryActiveMinutes'] = pd.cut(activity['VeryActiveMinutes'],bins = bins,labels = group_names) 
activity['VeryActiveMinutes'].unique()


# In[30]:


activity['VeryActiveMinutes'].dropna(axis=0,inplace=True)
print(activity['VeryActiveMinutes'])


# ### Assign a labels to our quality variable

# In[31]:


label_quality = LabelEncoder()


# In[32]:


activity['VeryActiveMinutes'] = label_quality.fit_transform(activity['VeryActiveMinutes'])


# In[33]:


# check VeryActiveMinutes label
activity.head(10)


# ### 0 means low active, 1 means very active. 

# In[34]:


activity['VeryActiveMinutes'].value_counts()


# In[35]:


# Double check there is no string " " left
# activity['VeryActiveMinutes'] = pd.to_numeric(activity['VeryActiveMinutes'])

len(activity.loc[activity['VeryActiveMinutes'] == 2])
activity.loc[activity['VeryActiveMinutes'] == 2]


# ### Remove Nan which means >1

# In[36]:


activity = activity.drop(activity[activity['VeryActiveMinutes']>1].index)
activity['VeryActiveMinutes'].value_counts()


# In[37]:


sns.countplot(activity['VeryActiveMinutes'])


# In[38]:


activity['VeryActiveMinutes'].unique()


# ### Separate the dataset as response variable and feature variables 

# In[39]:


# Drop unecessary columns


# In[40]:


activity.drop(['Id','ActivityDate'], axis=1,inplace=True)
print(activity)


# In[41]:


X = activity.drop('VeryActiveMinutes',axis=1)
y = activity['VeryActiveMinutes']


# ### Applying Standard scaling to get optimized result

# In[42]:


sc = StandardScaler()
X = sc.fit_transform(X)


# In[43]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y)


# ### 2.1.1 Working with Imbalance data 

# To solve imbalanced data is often by using SMOTE, Resampling and BalancedBaggingClassifier which depends on data. The technique of resampling is oversampling or downsampling the minority or majority class. Initially, I plan to use oversampling the minority class. After resampling the data we can get a balanced dataset for both majority and minority classes. So, when both classes have a similar number of records present in the dataset, 
# we can assume that the classifier will give equal importance to both classes.

# In[44]:


# Using the sklearn library’s resample() is shown below for illustration purposes. 
# Here, VeryActiveMinutes is our target variable. Let’s see the distribution of the classes in the target.


# In[45]:


sns.countplot(activity['VeryActiveMinutes'])


# In[46]:


# from sklearn.utils import resample

# create two different dataframe of majority and minority class

#activity_majority = X_train[(X_train["VeryActiveMinutes"] == 0)]
#activity_minority = X_train[(X_train["VeryActiveMinutes"] == 1)]

# Upsampling minority class

#activity_minority_ups = resample(activity_minority,
#                                 replace=True,
#                                 n_samples=787,
#                                 random_state=42
#                                ) 
# Combine majority calss with upsample minority class
#activity_upsample = pd.concat([activity_minority_ups,activity_majority])


# At the beginning, I try to oversample the minority but receive error "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices." Then, I realise that resample cannot apply the training data with integer or boolean arrays. Hence, I use SMOTE to balance the data.

# In[47]:


from imblearn.over_sampling import SMOTE

# Resampling the minority class. The strategy can be changed as required.

sm = SMOTE(sampling_strategy='minority', random_state=42)

# Fit the model to generate the data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y)

oversampled_X, oversampled_y = sm.fit_resample(X_train, y_train)



# In[48]:


oversample = pd.concat([pd.DataFrame(oversampled_X), pd.DataFrame(oversampled_y)], axis=1)
sns.countplot(oversample['VeryActiveMinutes'])


# ### Fibit dailyActivity with random forest 

# In[49]:


rfc = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc.fit(oversampled_X, oversampled_y)
pred_rfc = rfc.predict(X_test)


# In[50]:


# See model performed 
print(classification_report(y_test,pred_rfc))

# See confusion_matrix
print(confusion_matrix(y_test,pred_rfc))

from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc)

print("RandomForest gives the accuracy of:",cm*100)


# ### Fibit dailyActivity with SVM classifier 

# In[51]:


from sklearn import svm

clf = svm.SVC()
clf.fit(oversampled_X, oversampled_y)
pred_clf = clf.predict(X_test)

# see model performed 
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test,pred_clf))


# In[52]:


# See confusion_matrix
print(confusion_matrix( y_test, pred_clf))

from sklearn.metrics import accuracy_score
cm_svm = accuracy_score(y_test, pred_clf)

print("SVM gives the accuracy of:",cm_svm*100)


# ### Fibit dailyActivity with neural network

# In[53]:


# Although the dataset is not large but I would want to give a comparison with other two models. 

mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(oversampled_X, oversampled_y)
pred_mlpc=mlpc.predict(X_test)


# In[54]:


# see model performed 
print(classification_report(y_test,pred_mlpc))
print(confusion_matrix(y_test,pred_mlpc))


# In[55]:


from sklearn.metrics import accuracy_score
cm_mlpc = accuracy_score(pred_mlpc,y_test)

print("Neural Network gives the accuracy of:",cm_mlpc*100)


# ### Import Apple Watch series 2 data set

# In[56]:


df_aw = pd.read_csv('data_for_weka_aw.csv')
df_aw.head()


# In[57]:


# Identify Missing data
# Only the data type of activity_trimmed is object

df_aw.info()


# In[58]:


df_aw.drop(['Unnamed: 0'],axis=1,inplace=True)
df_aw


# In[59]:


df_aw.isnull().sum()


# In[60]:


# Check with data distribution in each column
# Most of the columns are gighly skewed or discrete. 

fig, axes = plt.subplots(nrows=1,ncols=4)
fig.set_size_inches(15, 4)
sns.histplot(df_aw["Applewatch.Steps_LE"][:], ax=axes[0],kde=True)
sns.histplot(df_aw["Applewatch.Calories_LE"][:],ax=axes[1], kde=True)
sns.histplot(df_aw["Applewatch.Distance_LE"][:],ax=axes[2], kde=True)
sns.histplot(df_aw["EntropyApplewatchHeartPerDay_LE"][:],ax=axes[3], kde=True)

plt.show()


# In[61]:


fig, axes = plt.subplots(nrows=1,ncols=4)
fig.set_size_inches(15, 4)

sns.histplot(df_aw["EntropyApplewatchStepsPerDay_LE"][:],ax=axes[0], kde=True)
sns.histplot(df_aw["RestingApplewatchHeartrate_LE"][:],ax=axes[1], kde=True)
sns.histplot(df_aw["CorrelationApplewatchHeartrateSteps_LE"][:],ax=axes[2], kde=True)
sns.histplot(df_aw["Applewatch.Heart_LE"][:],ax=axes[3], kde=True)


# In[62]:


fig, axes = plt.subplots(nrows=1,ncols=3)
fig.set_size_inches(15, 4)

sns.histplot(df_aw["ApplewatchIntensity_LE"][:],ax=axes[0], kde=True)
sns.histplot(df_aw["SDNormalizedApplewatchHR_LE"][:],ax=axes[1], kde=True)
sns.histplot(df_aw["ApplewatchStepsXDistance_LE"][:],ax=axes[2], kde=True)


# In[63]:


df_aw['activity_trimmed'].value_counts(dropna=False)


# In[64]:


sns.countplot(df_aw['activity_trimmed'])


# In[65]:


# The highest percentage of activity is lying 

import matplotlib.pyplot as plt

slices = df_aw['activity_trimmed'].value_counts()
labels = ['Lying','Running 7 METs','Running 5 METs','Running 3 METs','Sitting','Self Pace walk']
colors = ['lightblue','red','grey','green','yellow','pink']
explode = [0.1,0,0,0,0,0]

plt.pie(slices,autopct='%1.1f%%',labels= labels, colors= colors, explode=explode, shadow=True, startangle=90)
plt.title('Apple Watch 6 types of Activity')
plt.show()


# ### The Regplot shows there is a strong linear relationship between distance and steps.
# ### However the regression line is weighted down by high leverage noise in the data.

# In[66]:


plt.figure(figsize=(8,6))
sns.regplot(data = df_aw, x='Applewatch.Steps_LE', y='Applewatch.Distance_LE')
plt.title('Distance Vs. Steps')
plt.tight_layout()


# In[67]:


sns.distplot(df_aw['NormalizedApplewatchHeartrate_LE'])


# In[68]:


df_aw.hist(column='NormalizedApplewatchHeartrate_LE',bins = 25,density=True)
plt.show()


# In[69]:


# Map each value of Activity_trimmed into a number in order to encode

#df_aw['activity_trimmed'] = df_aw['activity_trimmed'].map({'Lying':1,'Running 7 METs':2,
#                                                        'Running 5 METs':3,'Running 3 METs':4,'Sitting':5,
#                                                           'Self Pace walk':6})
# 
#from sklearn import preprocessing 
#label_encoder = preprocessing.LabelEncoder()
#label_encoder.fit(df_aw['activity_trimmed'])
#label_encoder.transform(df_aw['activity_trimmed'])
#df_aw.head(10)


# After trying many times machine learing model, I detremine activity_trimmed should use one-hot encode instead of 
# label_encoder. There is no purpose of ranking or ordering for the six types of activity.

# In[70]:


# Boxplot for the skewed feature

import seaborn as sns
sns.set_theme(style="whitegrid")
aw = sns.boxplot(data = df_aw, x='Applewatch.Heart_LE',y='activity_trimmed')

#sns.choose_colorbrewer_palette("sequential")
plt.show()


# In[71]:


# A quick look at  the overview of scatter matrix graph

import plotly.express as px

fig = px.scatter_matrix(df_aw, dimensions=['EntropyApplewatchStepsPerDay_LE', 'RestingApplewatchHeartrate_LE', 
                                           'ApplewatchIntensity_LE', 'activity_trimmed', 
                                          'ApplewatchStepsXDistance_LE'], color='Applewatch.Heart_LE')
fig.show()


# In[72]:


# Use heatmap to check dataframe correlation  

plt.figure(figsize=(15,8))
sns.heatmap(df_aw.corr(),annot=True,fmt='.1f',cmap="RdBu_r")
plt.tight_layout()


# In[73]:


df_aw.columns


# In[74]:


# Split the data into dependant and indepandent variables

# Missing values have been taken care 
# Age, gender, height and weight are redundant variables.  
# 
# X = the columns of data that we will use to make classification
# y = the column of data that we want to predict


# In[75]:


predict_colum = [col for col in df_aw.columns if col not in ['age', 'gender', 'height', 'weight',
                                                             'activity_trimmed',
                                                             
                                                            
                                                            ]]

X = df_aw[predict_colum]

y = df_aw['activity_trimmed']


# In[76]:


X.dtypes


# In[77]:


y.unique()


# In[78]:


#Applying StandardScaler to get optimized result

from sklearn.preprocessing import StandardScaler

sc = StandardScaler() 
X= sc.fit_transform(X)
X[:10]


# In[79]:


# Split the training and testing data sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)


# In[80]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def calculate_inertia(k: int, df_aw: pd.DataFrame) -> float:
    """Calculate the intertia from a k means
    used to iterate over array of k values
    for elbow curve method 

    Args:
        k (int): K in KMeans
        df (pd.DataFrame): Data

    Returns:
        float: inertia
    """
 
    model = KMeans(n_clusters=k)
    model.fit(X_train)
    return model.inertia_


# ### Apply KMeans Clustering to see the correlation for each type of activity

# In[81]:


km = list(range(2,len(y_train.unique()) + 3))
ine = [calculate_inertia(k,X_train) for k in km]


# In[82]:


plt.figure(figsize = (10,7))
sns.pointplot( x=km, y=ine , linestyles='--', markers='o', scale=1.5, color='blue')
sns.set(font_scale=1.3)
plt.title('Inertia Vs K')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.tight_layout()


# ### The inertia elblow plot suggests that 4 would be an ideal number for k on the predictor space.

# In[83]:


model_km = KMeans(n_clusters = 4 , random_state=42)
model_km.fit_predict(X_train)
clusters = pd.Series(model_km.fit_predict(X)).rename('clusters',inplace= True)


# In[84]:


plt.figure(figsize = (13,10))
cross_tab = pd.crosstab(y_train,clusters,normalize=False)
cross_tab.values
row_sums = cross_tab.values.sum(axis=1)
proportions = cross_tab / row_sums.reshape((6,1))
sns.heatmap(proportions,annot=True,fmt='.2f')
plt.title('Proportions of Activities in clusters')
plt.tight_layout()


# ### Clusters has low correlation with each type of activity.

# ### 2.1.2 Compiling training data
# 
# ####  Apply Random Forest Classifier, SVM, XGBoost to determine which has the  hightest accurancy of each type of activity  

# The column of activity_trimmed does not natively support categorical data.
# My thinking is if we convert these categories to number 1,2,3,4,5 and 6, treated them like "continuous data", 
# then we would assume that 6 which means "Self Pace walk", is more similar to 5, which means "Sitting", then
# 4,3,2,1, 0 which means lying. In contrast, if we treat these type of activity like "categorial data", then we treat each one as a separate category that is no more or less similar to any of the other categories. They are equal weightage.Thus, the likelihood of people who do sitting with people who do lying is the same clustering with other activities, this is more reasonal approach. Thus, in this project, I apply one-hot encoding for trees to train the data. 

# In[85]:


# Two popular methods:
# ColumnTransform() (from scikit-learn) and get_dummies() (from pandas)
# Here I apply get_dummies


# ### Convert object data type into numeric variables with one-hot encoding

# In[86]:


df_aw_onehot= pd.get_dummies(df_aw['activity_trimmed'])  
df_aw_onehot

df_encode = pd.concat([df_aw[predict_colum],df_aw_onehot], axis = 1)  
df_encode                       # New data set


# ### Apply random forest on sitting 

# In[87]:


#Applying Standard scaling to get optimized result because the data set inside of the numbers range are not good

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = df_encode.drop(columns=['Sitting'],axis=1)

y = df_encode['Sitting']

X = sc.fit_transform(X)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[88]:


y.unique()


# In[89]:


# imbalanced data
y.value_counts()


# In[90]:


rfc_sit = RandomForestClassifier(max_depth=2)
rfc_sit.fit(X_train, y_train)

pred_rfc_sit = rfc_sit.predict(X_test)
pred_rfc_sit[:20]


# In[91]:


# See confusion_matrix report , the f1-score for 0 is 0.94, and 1 is 0 repestively. 
# Thus the FP, data is highly imbalanced
print(confusion_matrix(y_test,pred_rfc_sit))
print(classification_report(y_test, pred_rfc_sit))


# In[92]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc_sit)

print("RandomForest gives accuracy of the activity of sitting :",cm*100)


# ### Evaluate the model's performance measures the recall, precision and F1 score. Now the perofrmance is not good!

# ### Apply SVM on sitting

# In[93]:


from sklearn import svm

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)


# In[94]:


from sklearn.metrics import accuracy_score
cm_svm = accuracy_score(y_test, pred_clf)

print("SVM gives the accuracy of:",cm_svm*100)


# #### Check with the roc_auc_score still very hight

# In[95]:


y_pred_proba = rfc_sit.predict_proba(X_test)
y_pred_proba

roc_auc_score(y_test,y_pred_proba[:,1])


# In[96]:


fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:,1])
print(fpr, tpr, threshold)


# ### The AUC line almost flies over the box.
# ### We can certainly confirm that the data was highly imbalanced.

# In[97]:


auc1 = auc(fpr, tpr)
## Plot the result
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() 


# ### Apply SMOTE into binary classes

# In[98]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[99]:


rfc = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
# setting the training data
rfc.fit(X_res, y_res)
# Predicting on test
pred = rfc.predict(X_test)
pred[:5]

# See model performed 
print(classification_report(y_test,pred))


# In[100]:


# See confusion_matrix
print(confusion_matrix(y_test,pred))

from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred)

print("RandomForest gives the accuracy of sitting:",cm*100)


# In[101]:


# SVM Classifier
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5)
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=42)

#X_res, y_res = sm.fit_resample(X_train, y_train)

#from sklearn import svm

#clf = svm.SVC()
#clf.fit(X_res, y_res)
#pred_clf = clf.predict(X_test)


# In[102]:


# see model performed 
#print(classification_report(y_test, pred_clf))
#print(confusion_matrix(y_test,pred_clf))


# ### After many times training data with SVM classifier both on Apple and Fitbit data sets, I think SVM Classifier is not a good training model for this project. It's overfitting and in this case more often than not getting more data is the only way out. In the research method section, as I mentioned that there are some points may cause a bad performance on SVM method. It's good to know there are much more complex situation in the reality. I need to study further in the future. 

# ### Apply random forest on running 7 METs

# In[106]:


sc = StandardScaler()
X7 = df_encode.drop(columns=['Running 7 METs'],axis=1)

y7 = df_encode['Running 7 METs']

X7 = sc.fit_transform(X7)
#y = sc.fit_transform(y)

# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X7, y7, test_size = 0.2)# random_state = 42,stratify=y7


# In[107]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[108]:


rfc_7 = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_7.fit(X_res, y_res)
pred_rfc_7 = rfc_7.predict(X_test)

pred_rfc_7[:20]


# In[109]:


# See confusion_matrix
print(confusion_matrix(y_test,pred_rfc_7))


# In[110]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_7)

print("RandomForest gives accuracy of the activity of Running 7 METs:",cm*100)


# ### Apply random forest on lying

# In[111]:


sc = StandardScaler()
X_lying = df_encode.drop(columns=['Lying'],axis=1)

y_lying = df_encode['Lying']

X_lying = sc.fit_transform(X_lying)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X_lying, y_lying, test_size = 0.2)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
                                                    


# In[112]:


rfc_l = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_l.fit(X_res, y_res)
pred_rfc_l = rfc_l.predict(X_test)

pred_rfc_l[:20]


# In[113]:


# See confusion_matrix
print(confusion_matrix(y_test, pred_rfc_l))


# In[114]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_l)

print("RandomForest gives accuracy of the activity of Lying:",cm*100)


# ### Apply random forest on running 3 METs

# In[115]:


sc = StandardScaler()
X3 = df_encode.drop(columns=['Running 3 METs'],axis=1)

y3 = df_encode['Running 3 METs']

X3 = sc.fit_transform(X3)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.2)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[116]:


rfc_3 = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_3.fit(X_res, y_res)
pred_rfc_3 = rfc_3.predict(X_test)

pred_rfc_3[:20]


# In[117]:


# determin confusion_matrix
print(confusion_matrix(y_test,pred_rfc_3))


# In[118]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_3)

print("RandomForest gives accuracy of the activity of Running 3 METs:",cm*100)


# ### Apply random forest on running 5 METs

# In[119]:


sc = StandardScaler()
X5 = df_encode.drop(columns=['Running 5 METs'],axis=1)

y5 = df_encode['Running 5 METs']

X5 = sc.fit_transform(X5)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X5, y5, test_size=0.2, random_state=42, stratify=y5) 


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
                                                    


# In[120]:


rfc_5 = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_5.fit(X_res, y_res)
pred_rfc_5 = rfc_5.predict(X_test)

pred_rfc_5[:20]


# In[121]:


# See confusion_matrix
print(confusion_matrix(y_test,pred_rfc_5))


# In[122]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_5)

print("RandomForest gives accuracy of the activity of Running 5 METs:",cm*100)


# ### Apply random forest on self pace walk

# In[123]:


sc = StandardScaler()
Xw = df_encode.drop(columns=['Self Pace walk'],axis=1)

yw = df_encode['Self Pace walk']

Xw = sc.fit_transform(Xw)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xw, yw, test_size = 0.2, random_state = 42,stratify=yw)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[124]:


rfc_w = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_w.fit(X_res, y_res)
pred_rfc_w = rfc_w.predict(X_test)

pred_rfc_w[:10]


# In[125]:


# See confusion_matrix
print(confusion_matrix(y_test, pred_rfc_w))


# In[126]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_w)

print("RandomForest gives accuracy of the activity of Self Pace Walk:",cm*100)


# ## 2.1.3 Tune model hyperameters
# 
# ### Apply XGBoost on sitting

# In[127]:


import xgboost as xgb

df_encode 


# In[128]:


sc = StandardScaler()
Xs = df_encode.drop(columns=['Sitting'],axis=1)

ys = df_encode['Sitting']

Xs = sc.fit_transform(Xs)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.2, random_state = 42,stratify=ys)


# ### Optimize parameters using cross validation and GridSearch()

# In[129]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[130]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[131]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8, # using random subset of the data (80%), 90% are overfitting
                                colsample_bytree=0.5),  # using random features (50%) 
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}


# In[132]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=2,      # avoiding overfitting
                            reg_lambda=0,
                            scale_post_weight=1,
                            subsample=0.8,
                            dolsample_bytree=0.5,
                            )

#{'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}


# In[133]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.2, random_state = 42,stratify=ys)


# In[134]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 10 tress


# In[135]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm_sit= confusion_matrix(prediction,y_test)
print(cm_sit)

acc_score_sit = accuracy_score(prediction,y_test)*100

print("The accurancy score of XGBoost in Sitting: ",acc_score_sit)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Sitting','Not Sitting'])


# ### Apply XGBoost on lying

# In[136]:


sc = StandardScaler()
Xl = df_encode.drop(columns=['Lying'],axis=1)

yl = df_encode['Lying']

Xl = sc.fit_transform(Xl)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xl, yl, test_size = 0.2, random_state = 42,stratify=yl)


# In[137]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}

import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[138]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}


# In[139]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=3,
                            reg_lambda=10,          # low accurancy of 78.41 if reg_lambda': 0
                            scale_post_weight=2,
                            subsample=0.8,
                            dolsample_bytree=0.5,
                            )

# determin from second times optimise result to adjust parameters
#  {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}


# In[140]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xl, yl, test_size = 0.2, random_state = 42,stratify=yl)


# In[141]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 13 tress


# In[142]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score = accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Lying: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Lying','Not Lying'])


# ### Apply XGBoost on running 3METs

# In[143]:


sc = StandardScaler()
Xr3 = df_encode.drop(columns=['Running 3 METs'],axis=1)

yr3 = df_encode['Running 3 METs']

Xr3 = sc.fit_transform(Xr3)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr3, yr3, test_size = 0.2, random_state = 42,stratify=yr3)


# In[144]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[145]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 1.0, 'scale_pos_weight': 3}


# In[146]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=3,
                            reg_lambda=10,  # the accurancy is 84.23 if reg_lambda=10
                            scale_post_weight=1,
                            subsample=0.8,
                            dolsample_bytree=0.5,
                            )


# In[147]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr3, yr3, test_size = 0.2, random_state = 42,stratify=yr3)


# In[148]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 14 tress


# In[149]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction, y_test)
print(cm)

acc_score= accuracy_score(prediction, y_test)*100

print("The accuracy score of XGBoost in Running 3 METs: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Running 3 METs','Not Running 3 METs'])


# ### Apply XGBoost running 5 METs

# In[150]:


sc = StandardScaler()
Xr5 = df_encode.drop(columns=['Running 5 METs'],axis=1)

yr5 = df_encode['Running 5 METs']

Xr5 = sc.fit_transform(Xr5)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr5, yr5, test_size = 0.2, random_state = 42,stratify=yr5)


# In[151]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[152]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[153]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}


# In[154]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0,   
                            learn_rate=0.1,
                            max_depth=3,
                            reg_lambda=0,  
                            scale_post_weight=3,
                            subsample=0.8,
                            dolsample_bytree=0.5,
                            )
#{'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 3}

# the accurancy is 89.85 if {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3,
#'reg_lambda': 0, 'scale_pos_weight': 2}


# In[155]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr5, yr5, test_size = 0.2, random_state = 42,stratify=yr5)


# In[156]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 16 tress


# In[157]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score= accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Running 5 METs: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Running 5 METs','Not Running 5 METs'])


# ### Apply XGBoost on running 7METs

# In[158]:


sc = StandardScaler()
Xr7 = df_encode.drop(columns=['Running 7 METs'],axis=1)

yr7 = df_encode['Running 7 METs']

Xr7 = sc.fit_transform(Xr7)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr7, yr7, test_size = 0.2, random_state = 42,stratify=yr7)


# In[159]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[160]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[161]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 10.0, 'scale_pos_weight': 3}


# In[162]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=2,
                            reg_lambda=0,
                            scale_post_weight=2,
                            subsample=0.8,
                            dolsample_bytree=0.5,
                            )

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 10.0, 'scale_pos_weight': 3}


# In[163]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr7, yr7, test_size = 0.2, random_state = 42,stratify=yr7)


# In[164]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 27 tress


# In[165]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score= accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Running 7 METs: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Running 7 METs','Not Running 7 METs'])


# ### Apply XGBoost on self pace walk

# In[166]:


sc = StandardScaler()
Xpw = df_encode.drop(columns=['Self Pace walk'],axis=1)

ypw = df_encode['Self Pace walk']

Xpw = sc.fit_transform(Xpw)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xpw, ypw, test_size = 0.2, random_state = 42,stratify=ypw)


# In[167]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[168]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[169]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.4),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}


# In[170]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=2,
                            reg_lambda=0,
                            scale_post_weight=3,
                            subsample=0.8,
                            dolsample_bytree=0.4,
                            )

## {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}


# In[171]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xpw, ypw, test_size = 0.2, random_state = 42,stratify=ypw)


# In[172]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 10 tress


# In[173]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score= accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Self Pace Walk: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Self Pace Walk','Not Self Pace Walk'])                    


# ## Import Fitbit Charge HR2 dataset

# In[174]:


df_fb = pd.read_csv('data_for_weka_fb.csv')
df_fb.head()


# In[175]:


df_fb.info()


# In[176]:


df_fb.drop(['Unnamed: 0'],axis=1,inplace=True)
df_fb


# In[177]:


df_fb.isnull().sum()


# In[178]:


# Check with data distribution in each column
fig, axes = plt.subplots(nrows=1,ncols=4)

fig.set_size_inches(15, 4)
sns.histplot(df_fb["Fitbit.Steps_LE"][:], ax=axes[0],kde=True)
sns.histplot(df_fb["Fitbit.Heart_LE"][:],ax=axes[1], kde=True)
sns.histplot(df_fb["Fitbit.Calories_LE"][:],ax=axes[2], kde=True)
sns.histplot(df_fb["Fitbit.Distance_LE"][:],ax=axes[3], kde=True)


# In[179]:


fig, axes = plt.subplots(nrows=1,ncols=4)
fig.set_size_inches(15, 4)

sns.histplot(df_fb["EntropyFitbitHeartPerDay_LE"][:],ax=axes[0], kde=True)
sns.histplot(df_fb["EntropyFitbitStepsPerDay_LE"][:],ax=axes[1], kde=True)
sns.histplot(df_fb["RestingFitbitHeartrate_LE"][:],ax=axes[2], kde=True)
sns.histplot(df_fb["CorrelationFitbitHeartrateSteps_LE"][:],ax=axes[3], kde=True)


# In[180]:


fig, axes = plt.subplots(nrows=1,ncols=4)
fig.set_size_inches(15, 4)

sns.histplot(df_fb["NormalizedFitbitHeartrate_LE"][:],ax=axes[0], kde=True)
sns.histplot(df_fb["FitbitIntensity_LE"][:],ax=axes[1], kde=True)
sns.histplot(df_fb["SDNormalizedFitbitHR_LE"][:],ax=axes[2], kde=True)
sns.histplot(df_fb["FitbitStepsXDistance_LE"][:],ax=axes[3], kde=True)


# ### The polynomial regression shows there is no linear corelation between distance and steps.

# In[181]:


plt.figure(figsize=(8,6))
sns.regplot(data = df_fb, x="Fitbit.Steps_LE", y="Fitbit.Distance_LE",color='g',order=2)
plt.title('Distance Vs. Steps')
plt.tight_layout()


# In[182]:


df_fb['activity_trimmed'].value_counts()


# In[183]:


import matplotlib.pyplot as plt

slices = df_fb['activity_trimmed'].value_counts()
labels = ['Lying','Running 7 METs','Running 5 METs','Running 3 METs','Sitting','Self Pace walk']
colors = ['red','grey','lightblue','green','yellow','pink']
explode = [0.1,0,0,0,0,0]

plt.pie(slices,autopct='%1.1f%%',labels= labels, colors= colors, explode=explode, shadow=True, startangle=90)
plt.title('FitBit 6 types of Activity')
plt.show()


# In[184]:


sns.set_theme(style="whitegrid")

fb = sns.boxplot(data = df_fb, x="NormalizedFitbitHeartrate_LE",y="activity_trimmed")

plt.show()

# It smeems like lots of outliers for each type of activity. 
# I am not rushing to remove the outliers, I would like to check the correlation in heatmap
# and the KMeans clustering afterwards


# In[185]:


df_fb.hist(column='NormalizedFitbitHeartrate_LE',bins = 12,density=True)
plt.show()

# Visualising the density of normalizas heart rate


# In[186]:


# use heatmap to check dataframe correlation 

plt.figure(figsize=(15,8))
sns.heatmap(df_fb.corr(),annot=True,fmt='.1f',cmap="RdBu_r")
plt.tight_layout()


# In[187]:


df_fb.columns


# In[188]:


import plotly.express as px

fig = px.scatter_matrix(df_fb, dimensions=['Fitbit.Steps_LE', 'Fitbit.Distance_LE', 
                                           'FitbitIntensity_LE', 'activity_trimmed', 
                                          'SDNormalizedFitbitHR_LE'], color='RestingFitbitHeartrate_LE')
fig.show()


# In[189]:


# Reflect the AppleWatch dataset 
# After trying many times machine learing model, I think activity_trimmed should use one-hot encode instead of 
# label_encoder. Here, there is no purpose of ranking or ordering for the six types of activity.


# In[190]:


predict_colum2 = [col for col in df_fb.columns if col not in ['age', 'gender', 'height', 'weight',
                                                             'activity_trimmed'
                                                                                                                  
                                                            ]]

#X = df_aw.drop(columns=['age','gender','height','weight','activity_trimmed','Applewatch.Calories_LE'],axis=1)

X = df_fb[predict_colum2]

y = df_fb['activity_trimmed']


# In[191]:


X.dtypes


# In[192]:


y.unique()


# ### Inside the range of data are not good. Applying standard scaler to get optimized results.

# In[193]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler() 
X= sc.fit_transform(X)
X[:10]


# ### Split the training and testing data setsinto 80% and 20%.

# In[194]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)


# In[195]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def calculate_inertia(k: int, df_fb: pd.DataFrame) -> float:
    """Calculate the intertia from a k means
    used to iterate over array of k values
    for elbow curve method 

    Args:
        k (int): K in KMeans
        df (pd.DataFrame): Data

    Returns:
        float: inertia
    """
 
    model = KMeans(n_clusters=k)
    model.fit(X_train)
    return model.inertia_


# ### Apply K-Means Clustering

# In[196]:


km = list(range(2,len(y_train.unique()) + 3))
ine = [calculate_inertia(k,X_train) for k in km]


# In[197]:


plt.figure(figsize = (10,7))
sns.pointplot( x=km, y=ine , linestyles='--', markers='o', scale=1.5, color='green')
sns.set(font_scale=1.3)
plt.title('Inertia Vs K')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.tight_layout()


# ### The inertia elblow plot suggests that 4 would be an ideal number for k on the predictor space.

# In[198]:


model_km = KMeans(n_clusters = 4 , random_state=42)
model_km.fit_predict(X_train)
clusters = pd.Series(model_km.fit_predict(X)).rename('clusters',inplace= True)


# In[199]:


plt.figure(figsize = (10,8))
cross_tab = pd.crosstab(y_train,clusters,normalize=False)
cross_tab.values
row_sums = cross_tab.values.sum(axis=1)
proportions = cross_tab / row_sums.reshape((6,1))
sns.heatmap(proportions,annot=True,fmt='.2f')
plt.title('Fibit Proportions of Activities in Clusters')
plt.tight_layout()


# ### Cluster 0 is has more strong correction with each activity excepts Running 7 METs. 

# ### Apply fandom forest and XGBoost in each type of activity

# #### Convert activity_trimmed into one-hot encoding

# In[200]:


df_fb_onehot= pd.get_dummies(df_fb['activity_trimmed'])  
df_fb_onehot

df_encode2 = pd.concat([df_fb[predict_colum2],df_fb_onehot], axis = 1)  
df_encode2                       # New data set


# ### Apply random forest on sitting

# In[201]:


#Applying Standard scaling to get optimized result because the data set inside of the numbers range are not good

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = df_encode2.drop(columns=['Sitting'],axis=1)

y = df_encode2['Sitting']

X = sc.fit_transform(X)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[202]:


rfc_sit = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)

rfc_sit.fit(X_res, y_res)
pred_rfc_sit = rfc_sit.predict(X_test)

pred_rfc_sit[:10]


# In[203]:


# See Confusion_Matrix
print(confusion_matrix(y_test,pred_rfc_sit))

from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc_sit)
print(classification_report(y_test,pred_rfc_sit))

print("RandomForest gives accuracy of the activity of Sitting :",cm*100)


# ### Apply random forest on running 7 METs

# In[204]:


sc = StandardScaler()
X7 = df_encode2.drop(columns=['Running 7 METs'],axis=1)

y7 = df_encode2['Running 7 METs']

X7 = sc.fit_transform(X7)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X7, y7, test_size = 0.2, random_state = 42,stratify=y7)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[205]:


rfc_7 = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_7.fit(X_res, y_res)
pred_rfc_7 = rfc_7.predict(X_test)

pred_rfc_7[:10]


# In[206]:


# See confusion_matrix
print(confusion_matrix(y_test,pred_rfc_7))

from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_7)

print(classification_report(y_test,pred_rfc_7))
print("RandomForest gives accuracy of the activity of Running 7 METs:",cm*100)


# ### Apply random forest on lying

# In[207]:


sc = StandardScaler()
X_l = df_encode2.drop(columns=['Lying'],axis=1)

y_l = df_encode2['Lying']

X_l = sc.fit_transform(X_l)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, test_size = 0.2, 
                                                    random_state = 42,stratify=y_l)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[208]:


rfc_l = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_l.fit(X_res, y_res)
pred_rfc_l = rfc_l.predict(X_test)

pred_rfc_l[:10]


# In[209]:


# See Confusion_Matrix
print(confusion_matrix(y_test, pred_rfc_l))


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_l)
print(classification_report(y_test,pred_rfc_l))

print("RandomForest gives accuracy of the activity of Lying:",cm*100)


# ### Apply random forest running 3 METs

# In[210]:


sc = StandardScaler()
X3s = df_encode2.drop(columns=['Running 3 METs'],axis=1)

y3s = df_encode2['Running 3 METs']

X3s = sc.fit_transform(X3s)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(X3s, y3s, test_size = 0.2, random_state = 42,stratify=y3s)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[211]:


rfc_3 = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_3.fit(X_res, y_res)
pred_rfc_3 = rfc_3.predict(X_test)

pred_rfc_3[:10]


# In[212]:


# determin confusion_matrix
print(confusion_matrix(y_test,pred_rfc_3))


# In[213]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_3)
print(classification_report(y_test,pred_rfc_3))

print("RandomForest gives accuracy of the activity of Running 3 METs:",cm*100)


# ### Apply random forest running 5 METs

# In[214]:


sc = StandardScaler()
Xr5 = df_encode2.drop(columns=['Running 5 METs'],axis=1)

yr5 = df_encode2['Running 5 METs']

Xr5 = sc.fit_transform(Xr5)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr5, yr5, test_size = 0.2, 
                                                    random_state = 42,stratify=yr5)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[215]:


rfc_5 = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_5.fit(X_res, y_res)
pred_rfc_5 = rfc_5.predict(X_test)

pred_rfc_5[:10]


# In[216]:


# See confusion_matrix
print(confusion_matrix(y_test,pred_rfc_5))

from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_5)
print(classification_report(y_test,pred_rfc_5))

print("RandomForest gives accuracy of the activity of Running 5 METs:",cm*100)


# ### Apply random forest self pace walk

# In[217]:


sc = StandardScaler()
Xp = df_encode2.drop(columns=['Self Pace walk'],axis=1)

yp = df_encode2['Self Pace walk']

Xp = sc.fit_transform(Xp)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xp, yp, test_size = 0.2, random_state = 42,stratify=y)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[218]:


rfc_w = RandomForestClassifier(max_depth=2, n_jobs=1, max_features=1)
rfc_w.fit(X_res, y_res)
pred_rfc_w = rfc_w.predict(X_test)

pred_rfc_w[:10]


# In[219]:


# See confusion_matrix
print(confusion_matrix(y_test, pred_rfc_w))

from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test,pred_rfc_w)
print(classification_report(y_test,pred_rfc_w))

print("RandomForest gives accuracy of the activity of Self Pace Walk:",cm*100)


# ### Apply XGBoost on sitting 

# In[220]:


sc = StandardScaler()
Xs = df_encode2.drop(columns=['Sitting'],axis=1)

ys = df_encode2['Sitting']

Xs = sc.fit_transform(Xs)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.2)


# In[221]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[222]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[223]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.4),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)

# {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 1.0, 'scale_pos_weight': 2}


# In[224]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=2,
                            reg_lambda=0,
                            scale_post_weight=1,
                            subsample=0.8,
                            dolsample_bytree=0.4,
                            )

#{'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}


# In[225]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.2)


# In[226]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 24 tress


# In[227]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm_sit= confusion_matrix(prediction,y_test)
print(cm_sit)

acc_score_sit = accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Sitting: ",acc_score_sit)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Sitting','Not Sitting'])


# ### Apply XGBoost on lying

# In[228]:


sc = StandardScaler()
Xl = df_encode2.drop(columns=['Lying'],axis=1)

yl = df_encode2['Lying']

Xl = sc.fit_transform(Xl)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xl, yl, test_size = 0.2, random_state = 42,stratify=yl)


# In[229]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}

import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[230]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.4),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)


# In[231]:


clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=2,
                            reg_lambda=0,          
                            scale_post_weight=1,
                            subsample=0.8,
                            dolsample_bytree=0.4,
                            )
# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}


# In[232]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xl, yl, test_size = 0.2)


# In[233]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 21 tress


# In[234]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score = accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Lying: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Lying','Not Lying'])


# ### Apply XGBoost on running 3 METs

# In[235]:


sc = StandardScaler()
Xr3 = df_encode2.drop(columns=['Running 3 METs'],axis=1)

yr3 = df_encode2['Running 3 METs']

Xr3 = sc.fit_transform(Xr3)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr3, yr3, test_size = 0.3)


# In[236]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[237]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.8,
                                colsample_bytree=0.4),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)


# In[238]:


# {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=3,   #  trying to increase the test size, result are similar
                            reg_lambda=10,  
                            scale_post_weight=2,
                            subsample=0.8,
                            dolsample_bytree=0.4,
                            )


# In[239]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr3, yr3, test_size = 0.2)


# In[240]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 11 tress


# In[241]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction, y_test)
print(cm)

acc_score= accuracy_score(prediction, y_test)*100

print("The accuracy score of XGBoost in Running 3 METs: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Running 3 METs','Not Running 3 METs'])


# ### Apply XGBoost running 5 METs

# In[242]:


sc = StandardScaler()
Xr5 = df_encode2.drop(columns=['Running 5 METs'],axis=1)

yr5 = df_encode2['Running 5 METs']

Xr5 = sc.fit_transform(Xr5)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr5, yr5, test_size = 0.3)


# In[243]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[244]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[245]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)


# In[246]:


# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 2}

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,   
                            learn_rate=0.1,
                            max_depth=2,
                            reg_lambda=0,  
                            scale_post_weight=1,
                            subsample=0.9,
                            dolsample_bytree=0.5)


# In[247]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr5, yr5, test_size = 0.3)


# In[248]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 29 tress


# In[249]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score= accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Running 5 METs: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Running 5 METs','Not Running 5 METs'])


# ### Apply XGBoost on running 7 METs

# In[250]:


sc = StandardScaler()
Xr7 = df_encode2.drop(columns=['Running 7 METs'],axis=1)

yr7 = df_encode2['Running 7 METs']

Xr7 = sc.fit_transform(Xr7)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr7, yr7, test_size = 0.2, random_state = 42,stratify=yr7)


# In[251]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[252]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[253]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)


# In[254]:


# {'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.25,
                            learn_rate=0.1,
                            max_depth=2,
                            reg_lambda=0,
                            scale_post_weight=1,
                            subsample=0.9,
                            dolsample_bytree=0.5,
                            )


# In[255]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xr7, yr7, test_size = 0.3)


# In[256]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=10, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 33 tress


# In[257]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score= accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Running 7 METs: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Running 7 METs','Not Running 7 METs'])


# ### Apply XGBoost on self pace walk

# In[258]:


sc = StandardScaler()
Xpw = df_encode2.drop(columns=['Self Pace walk'],axis=1)

ypw = df_encode2['Self Pace walk']

Xpw = sc.fit_transform(Xpw)


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xpw, ypw, test_size = 0.3)


# In[259]:


# Optimize Parameters using Cross Validation and GridSearch()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,2,3]   # balance positive and negative weight
}


# In[260]:


import xgboost as xgb
from xgboost import XGBClassifier

clf_xgb = XGBClassifier

eval_set=[(X_test,y_test)]


# In[261]:


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid = param_grid,
    scoring = 'roc_auc',
    verbose= 0,
    n_jobs = 10,
    cv = 3    
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False                   
)

print(optimal_params.best_params_)


# In[262]:


# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'reg_lambda': 0, 'scale_pos_weight': 1}

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic', 
                            gamma=0.1,
                            learn_rate=0.1,
                            max_depth=1,
                            reg_lambda=10,
                            scale_post_weight=1,
                            subsample=0.9,
                            dolsample_bytree=0.5,
                            )


# In[263]:


# Train and Test splitting of data
# Split the data into 80% of training and 20% of testing 
X_train, X_test, y_train, y_test = train_test_split(Xpw, ypw, test_size = 0.3)


# In[264]:


clf_xgb.fit(X_train,
            y_train,
            verbose=True, 
            early_stopping_rounds=5, 
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

clf_xgb.score(X_test, y_test)

# model builds 36 tress


# In[265]:


from sklearn.metrics import plot_confusion_matrix


prediction = clf_xgb.predict(X_test)

cm= confusion_matrix(prediction,y_test)
print(cm)

acc_score= accuracy_score(prediction,y_test)*100

print("The accuracy score of XGBoost in Self Pace Walk: ",acc_score)

plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=['Self Pace Walk','Not Self Pace Walk'])                    


# # 3. Conclusion
# 
# ## 3.1 Summary of outcomes 
# 
# - The classification accuracy of dailyActivity dataset for Fibit gives 80% in random forest, 93% in SVM models and 95% in neural network separately. The best predicted model is random forest. 
# - The classification accuracy of Apple watch series2 with random forest gives 80.73% for self pace walk, 85.95% for sitting and 93.3% for running 3METs. The overall prediction of each feature (column) shows a good performance. 
# - The classification accuracy of Apple watch series2 with XGBoost gives 75.63% for self pace walk, 85.95% for sitting and 89.20% for running 5 METs. The highest classification accuracy is 89.61% for running 7METs.   
# - For random forest model, the highest classification accuracy of Fibit Charge HR2  gives 93.67% for running 5 METs, 85.24% for self pace walk and 86.59% for running 3 METs.  
# - XGBoost model of Fibit Charge HR2 gives very high accuracy from 84.28% for running 3 METs to 98.85% for self pace walk. 
# - Fandom forest model gives the average of 84% for Apple watch and for 89% Fibit Charge HR2. The accuracy will change by each time running the model.
# - XGBoost model gives the average of 89% for Apple watch and for 96% Fibit Charge HR2. 
# 
# With random forest and XGBoost model training, the conclusion of this project can demonstrate that smart wearable is able to predict user physical activities. 
# 
# 
# ## 3.2 Summary of findings and reflection
# 
# In the model training, I was struggling to deal with different machine learning results back-and-forth countless. I reflect the reasons are mainly because the lack experiences of imbalanced data, binary classification and types of classification task in machine learning. When the training model results give very highly accuracy, the measurement from Receiver Operating Characteristic (ROC) curve, confusion matrix [11] of recall (or sensitivity) , precision ( the true positive / true positive + false negative) and F1 score should be considered carefully in statistic level. At the begining of model training, I used all features (columns of six type activities) to train the model in random forest. From Rok and Lara(2013)[10] mentioned the high-dimensionality may affect each type of classifier in a different way. A general remark is that large discrepancies between training data and true population values are more likely to occur in the minority class, which has a larger sampling variability: therefore, the classifiers are often trained on data that do not represent well the minority class. The high-dimensionality contributes to this problem as extreme values are not exceptional when thousands of variables are considered. As my purpose in this project is trying to predict how commercial wearable can detect user physical movements. Therefore, I adopt Synthetic Minority Oversampling Technique (SMOTE) to resample the training set in binary classification and reduce the impact of imbalanced data. 
# 
# In addition, in the classification machine learning, tuning parameters are very important since they often control the complexity of the model and thus also affect any variance-base trade-off that can be made. For example, for random forest, the setting of max_depth, max_feature and n_jobs would give fluctuating 12-15% accuracy. To sum up, there are much more criteria and techniques required to learn. For me, it is the learning process. It is good to know and learn from failure. 
# 
# 
# ## 3.3 Summary of approach 
# 
# Users has disease patterns and require remote monitoring detection. Smart wearables have the potential to offer a minimally- obtrusive telemedicine platform for individuals health services that are easily, time-saving, cost-effective and better quality for the patient and citizen. By using the cloud service and the spread of Internet-of Things(IoT) in domains like health monitoring, users can conveniently wear smart devices with health minatory system at home, hospital or gym. However, I may consider the security, privacy and energy efficiency. The devices can collect real-time data related to millions users’ health and behavior which have great commercial and social value. Secondly, when devices upload data to the cloud may easily make the system vulnerable to attacks and data leakage. 
# 
# In 2021, Fitbit was officially part of Google. Rick Osterloh, Google’s Senior Vice President, Devices & Services, said “Google will continue to protect Fitbit users’ privacy and has made a series of binding commitments with global regulators, confirming that Fitbit users’ health and wellness data won’t be used for Google ads and this data will be kept separate from other Google ad data”. I google how Fitbit users react with privacy and securty. The relative results of how to delete/ erase your Fitbit data are very popular. Surprisingly, it will take 90 days to totally remove personal information from any subscription to backup system.  

# # 4. References and resources 
# 
# ### Reference
# * [1]Benedikt Schnell, Patrick Moder, Hans Ethm, Marcel Konstantinov and Mahmoud Ismail (2022)Challenges in Smart Health Applications Using Wearable Medical Internet-of-Things—A Review. 
# * [2]Behav Ther (2020) Supervised machine learning: A brief primer.
# * [3]Rebecca Roelofs, Sara Fridovich-Keil, John Miller, Vaishaal Shankar, Moritz Hardt, Benjamin Recht, Ludwig Schmidt (2019)A Meta-Analysis of Overfitting in Machine Learning. 
# * [4]Garvesh Raskutti, Martin J. Wainwright, Bin Yu (2014)Early Stopping and Non-parametric Regression: An Optimal Data-dependent Stopping Rule.
# * [5]Fuller, Daniel (2020) Replication Data for: Using machine learning methods to predict physical activity types with Apple Watch and Fitbit data using indirect calorimetry as the criterion.
# * [6]Harsurinder Kaur, Husanbir Singh Pannu, Avleen Kaur Malhi (2020)A Systematic Review on Imbalanced Data Challenges in Machine Learning: Applications and Solutions.
# * [7]Corinna Cortes and Vladimir Vapnik (1995)Support-Vector Networks. 
# * [8]Yuan-HaiShao, Wei-JieChen, Nai-YangDeng (2014)Nonparallel hyperplane support vector machine for binary classification problems
# * [9]Tianqi Chen, Carlos Guestrin (2016)XGBoost: A Scalable Tree Boosting System, p. 785-794.
# * [10]Blagus Rok, Lusa Lara (2013) SMOTE for high-dimensional class-imbalanced data.
# * [11]Zheng, Xiaoru (2020)SMOTE Variants for Imbalanced Binary Classification: Heart Disease Prediction. p.25
# * [12]Usama Fayyad, Gregory Piatetsky-Shapiro, Padhraic Smyth (1996)From Data Mining to Knowledge Discovery in Databases. 
# ### Resources
# Data Source
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZS2Z2J&version=1.0
# 
# Imbalanced data classification
# * Cheng Zhang, Yufei Chen, Xianhui Liu, Xiaodong Zhao (2017) Abstention-SMOTE: An over-sampling approach for imbalanced data classification. 
# * https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# 
# Wearables market share by IDC https://www.idc.com/promo/wearablevendor
# 
# Random Forest hyperparameters tunning 
# * https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
# * https://www.quora.com/Why-is-the-prediction-accuracy-of-without-tuning-Random-Forest-algorithm-greater-than-with-tuning-Random-Forest-algorithm
# 
# Overfitting and when to use SVM  
# * https://www.quora.com/How-can-I-avoid-over-fitting-in-SVM
# * https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-simple-binary-svm-classifier-with-python-and-scikit-learn.md
# * https://medium.com/swlh/support-vector-machines-algorithm-explained-60b7448b2f3e
# * https://www.quora.com/For-what-kind-of-classification-problems-is-SVM-a-bad-approach
# 
# AUC ROC curve 
# https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




