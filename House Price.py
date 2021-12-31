#!/usr/bin/env python
# coding: utf-8

# #This Dataset is a house sale prices dataset utilsied for a study - Author: Amrit Kaur, 19160239

# # IMPORTING LIBRARIES

# In[1]:


#Utilised for multidimentional arrays and matrix data structures (Linear Algebra Library)
import numpy as np


# In[2]:


#Utilised for analyses of data and manupilation tool, CVS files
import pandas as pd


# In[3]:


#pylot is a function in visualisation package Matplotlib. Manipulate elements of a figure(plotting area, lables)
import matplotlib.pyplot as plt


# In[4]:


#Utilised for data visualization and exploratory data analysis. Built on top of matplot
import seaborn as sns


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


#Transforms data, so its distribution will have a mean value 0 and standard deviation of 1
from sklearn.preprocessing import StandardScaler


# In[7]:


#Utilised to standardize a dataset along any axis
from sklearn import preprocessing


# In[8]:


#Utilised for accuracy classification score
from sklearn.metrics import confusion_matrix, accuracy_score


# In[9]:


from sklearn.model_selection import GridSearchCV


# In[10]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[11]:


#Importing the dataset and reading the datatset using 'pd.read_csv' command
House = pd.read_csv("train.csv")


# In[12]:


#Returns the first 5 rows for the object based on position.
House.head()


# In[13]:


#.index describe index for Dataset 
House.index


# In[14]:


#Returns number of rows, columns
House.shape


# In[15]:


#Returns columns in dataset
House.columns


# In[16]:


#Describes the dataset
House.describe()


# In[17]:


# Includes all categorical Variable
House.describe(include='all')


# In[18]:


#covariance - Measure the relationship between the returns on two assets
House.cov()


# In[19]:


#Utilised to find correlation of all columns in the dataset frame.
House.corr()


# # Handling missing values

# In[20]:


#Checking if there is any missing values on the Dataset
House.isnull().values.any()


# In[21]:


#Checking the total number of missing values on the Dataset
House.isnull().sum()


# In[22]:


#Determine columns with the missing values and separating them from other columns
House.loc[:, House.isnull().any()].columns


# In[23]:


House.dtypes 


# In[24]:


#Returns value count of each datatype ;
House.dtypes.value_counts()


# In[25]:


#Get informations on each columns values count and data type  
House.info()


# # Label Encoding Technique 

# In[26]:


#Converting Categorical data into numerical data
StringTransfer = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
for feature in StringTransfer:
    le = preprocessing.LabelEncoder()
    intFeature = House[feature]
    le.fit(intFeature)
    House[feature]= le.transform(intFeature)
print (House)


# In[27]:


House.mean()


# # Filling missing values

# In[28]:


#Filling missing values with the mean or mode value depending on datatype
House['LotFrontage'].fillna(House['LotFrontage'].mean(), inplace=True)                 #Numerical Datatype
House['Alley'].fillna(House['Alley'].mode(), inplace = True)                           #Categorical Datatype
House['MasVnrType'].fillna(House['MasVnrType'].mode(), inplace = True)                 #Categorical Datatype
House['MasVnrArea'].fillna(House['MasVnrArea'].mean(), inplace = True)                 #Numerical Datatype
House['BsmtQual'].fillna(House['BsmtQual'].mode(), inplace = True)                     #Categorical Datatype
House['BsmtCond'].fillna(House['BsmtCond'].mode(), inplace = True)                     #Categorical Datatype
House['BsmtExposure'].fillna(House['BsmtExposure'].mode(), inplace = True)             #Categorical Datatype
House['BsmtFinType1'].fillna(House['BsmtFinType1'].mode(), inplace = True)             #Categorical Datatype
House['BsmtFinType2'].fillna(House['BsmtFinType2'].mode(), inplace = True)             #Categorical Datatype
House['Electrical'].fillna(House['Electrical'].mode(), inplace = True)                 #Categorical Datatype
House['FireplaceQu'].fillna(House['FireplaceQu'].mode(), inplace = True)               #Categorical Datatype
House['GarageType'].fillna(House['GarageType'].mode(), inplace = True)                 #Categorical Datatype
House['GarageYrBlt'].fillna(House['GarageYrBlt'].mean(), inplace = True)               #Numerical Datatype
House['GarageFinish'].fillna(House['GarageFinish'].mode(), inplace = True)             #Categorical Datatype
House['GarageQual'].fillna(House['GarageQual'].mode(), inplace = True)                 #Categorical Datatype
House['GarageCond'].fillna(House['GarageCond'].mode(), inplace = True)                 #Categorical Datatype
House['PoolQC'].fillna(House['PoolQC'].mode(), inplace = True)                         #Categorical Datatype
House['Fence'].fillna(House['Fence'].mode(), inplace = True)                           #Categorical Datatype
House['MiscFeature'].fillna(House['MiscFeature'].mode(), inplace = True)               #Categorical Datatype


# In[29]:


#Dropping columns that have more than 30% missing values
House = House.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)


# In[30]:


#Converting float data into integer data
House = House.astype({"LotFrontage":'int', "MasVnrArea":'int', "GarageYrBlt":'int'})
print(House)


# In[31]:


#checking if all the missing values have been replaced. If the outcome is 'False' then all missing values have been replaced.
House.isnull().values.any()


# In[32]:


correlation = House.corr()
cor_target = abs(correlation["SalePrice"])
relevant_feet = cor_target[cor_target>0.5]
relevant_feet


# # Feature Engineering

# In[33]:


#Total sum of living space consisted of floors, garage and external spaces.
Livingtotalsq = House['TotalBsmtSF'] + House['1stFlrSF'] + House['2ndFlrSF'] + House['GarageArea'] + House['WoodDeckSF'] + House['OpenPorchSF']
House['LivingTotalSF'] = Livingtotalsq

#Dividing Living Area by LotArea
House['PercentSQtoLot'] = House['LivingTotalSF'] / House['LotArea']

#Total sum of all bathrooms including full and half through the entire building
House['TotalBaths'] = House['BsmtFullBath'] + House['BsmtHalfBath'] + House['HalfBath'] + House['FullBath']

#Percentage of the total number of bedrooms
House['PercentBedrmtoRooms'] = House['BedroomAbvGr'] / House['TotRmsAbvGrd']

# Number of years since last remodel
House['YearSinceRemodel'] = 2016 - ((House['YearRemodAdd'] - House['YearBuilt']) + House['YearBuilt'])


# In[34]:


#Barplot representing Total Square Footage with Overall Condition Rating
sns.barplot(x="OverallCond", y="LivingTotalSF", data=House)
plt.title("Total Square Footage with Overall Condition Rating")
plt.show()


# In[35]:


#lmplot representing relationship between Sale Price and Total Square Footage
sns.lmplot(x="LivingTotalSF", y="SalePrice", data=House)
plt.title("Relationship between Sale Price and Total Square Footage")
plt.xlim(0,)
plt.ylim(0,)
plt.show()


# In[36]:


#Representing Total number of rooms vs total number of bathrooms
ax = sns.barplot(x="TotRmsAbvGrd", y="TotalBaths",data=House)
plt.title("Total number of Rooms vs Total number of Bathrooms")
plt.show()


# In[37]:


#Representing Sale Month
sns.swarmplot(x="MoSold", y="SalePrice", data=House)
plt.title("Sale Price by Month")
plt.show()

sns.kdeplot(House['MoSold']);
plt.title("Distribution of Month Sold")
plt.xlim(1,12)
plt.show()


# # Data Visualisation

# In[38]:


#Linear Correlation between Above Grade Square feet and the sale price
sns.lmplot(x="GrLivArea", y="SalePrice", data=House);
plt.title("Linear Regression of Above Grade Square Feet and Sale Price")
plt.ylim(0,)
plt.show()


# In[39]:


#Linear Correlation between First floor Square feet and the sale price
sns.lmplot(x="1stFlrSF", y="SalePrice", data=House);
plt.title("Linear Regression of First Floor Square Feet and Sale Price")
plt.ylim(0,)
plt.show()


# In[40]:


#Converting series to dataset so it can be sorted
Datacorrelation = House.corr()['SalePrice']
#Correct column label from SalePrice to correlation
Datacorrelation = pd.DataFrame(Datacorrelation)
#Sorting correlation
Datacorrelation.columns = ["Correlation"]

Data_correlation = Datacorrelation.sort_values(by=['Correlation'], ascending=False)
Data_correlation.head(15)


# In[41]:


#Correlation between each feature and the Sale Price
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(House[cols], size = 2.5)
plt.show();


# In[42]:


#Correlation heatmap
corr = House.corr()
sns.heatmap(corr, annot = True)
plt.show()


# In[43]:


#Producing a heatmap
fig, ax = plt.subplots(figsize=(120,50))
sns.heatmap(House.corr(), cmap='viridis', annot=True)


# In[44]:


#Diving the Dataset in numerical and categorical
Numerical_coloumns = [col for col in House.columns if House[col].dtype in ['int64','float64']]
Categorical_columns = [col for col in House.columns if House[col].dtype == 'object']
a = House['SalePrice']
Numerical_Dataset = House[Numerical_coloumns]
Categorical_Dataset = House[Categorical_columns]


# In[45]:


Numerical_Dataset.dropna(axis=1, inplace=True)
Numerical_Dataset.nunique()


# In[46]:


Categorical_Dataset.dropna(axis=1, inplace=True)
Categorical_Dataset.nunique()


# In[47]:


#Printing 15 unique values from the numerical dataset columns
columns = [col for col in Numerical_Dataset.columns if Numerical_Dataset[col].nunique() > 15]
columns.remove('SalePrice')
characteristic = Numerical_Dataset[columns]


# In[48]:


sns.pairplot(characteristic)


# In[49]:


for idx, col in enumerate(characteristic.columns):
    plt.figure(idx, figsize=(5,5))
    sns.relplot(x=col, y=a, kind="scatter", data=characteristic)
    plt.show


# In[50]:


cols = [col for col in Numerical_Dataset.columns if Numerical_Dataset[col].nunique() <= 15 ] 

distribution_feature = Numerical_Dataset[cols]
distribution_feature.head()


# In[51]:


for idx, col in enumerate(distribution_feature.columns):
    plt.figure(idx, figsize=(5,5))
    sns.stripplot(x=col , y=a , data=distribution_feature)
    plt.show


# In[52]:


#Represting Sale Price occurence using histogram 
House['SalePrice'].hist(bins=50)


# # Split and Train Data

# In[53]:


House.head()


# In[54]:


House.tail()


# In[55]:


#Selecting feautures that have the value greater than 0.25 
X = House.loc[:,['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', 'GarageYrBlt', 'BsmtFinSF1', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF', 'HalfBath', 'GarageYrBlt', 'LotArea', 'BsmtFullBath', 'LotFrontage', 'BsmtUnfSF']]
y = House['SalePrice']


# In[56]:


#sc_X = StandardScaler()
#sc_X = sc_X.fit_transform(X)
#print(sc_X)


# In[57]:


X.head()


# In[58]:


House['SalePrice'].value_counts()


# In[59]:


X.shape


# In[60]:


y.shape


# In[61]:


X.isnull().sum()


# In[62]:


#Splitting and training the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[63]:


#Counting the value on y
print(y.value_counts())


# In[64]:


#Printing unique values
import numpy as np
np.unique(y)


# In[65]:


#Standardization of Data 
scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[66]:


#Returns the X value
X_train


# In[67]:


House.head()


# # Model Evaluation

# In[68]:


#Random forest regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

random_forest_regressor = RandomForestRegressor(random_state=1)
random_forest_regressor.fit(X_train, y_train)
prediction = random_forest_regressor.predict(X_test)
#accuracy=random_forest_regressor.score(X_test,y_test)
#accuracy=accuracy*100
#accuracy = float("{0:.4f}".format(accuracy))
#print(accuracy)
mean_squared_error(y_test, prediction)


# In[69]:


r2_score(y_test, prediction)


# In[70]:


max_error(y_test, prediction)


# In[71]:


explained_variance_score(y_test, prediction)


# In[72]:


#Initialising list of parameters
param_grid = { 
    "n_estimators"      : [10,20,30],
    "max_features"      : ["auto", "sqrt", "log2"],
    "min_samples_split" : [2,4,8],
    "bootstrap": [True, False],
}


# In[73]:


random_forest_regression =  RandomForestRegressor()


# In[74]:


gridRandomForest = GridSearchCV(random_forest_regression, param_grid, cv = 2)


# In[75]:


#Fitting the data to find the best parameter
gridRandomForest.fit(X_train, y_train)


# In[76]:


gridRandomForest.best_params_


# In[77]:


gridRandomForest = RandomForestRegressor(max_features ='auto')


# In[78]:


gridRandomForest.fit(X_train, y_train)
RFtunedprediction = gridRandomForest.predict(X_test)


# In[79]:


plt.figure(figsize=(10,10))
plt.scatter(y_test, prediction)
plt.yscale('log')
plt.xscale('log')

p1 = max(max(prediction), max(y_test))
p2 = min(min(prediction), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize = 12)
plt.ylabel('Predictions', fontsize = 12)
plt.axis('equal')
plt.show


# In[80]:


prediction.shape


# In[81]:


y_test.shape


# In[82]:


#Ridge regression Model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

ridgereg = Ridge(normalize=True)
ridgereg.fit(X_train, y_train)
ridgereg_prediction = ridgereg.predict(X_test)

mean_squared_error(y_test, ridgereg_prediction)


# In[83]:


r2_score(y_test, ridgereg_prediction)


# In[84]:


max_error(y_test, ridgereg_prediction)


# In[85]:


explained_variance_score(y_test, prediction)


# In[86]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge()
from sklearn.model_selection import GridSearchCV
params_Ridge = {'alpha': [1,0.1,0.01,0.001,0.0001,0] , "fit_intercept": [True, False], "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
Ridge_GS = GridSearchCV(ridge_reg, param_grid=params_Ridge, n_jobs=-1)
Ridge_GS.fit(X_train,y_train)
#Predicting best parameters for Ridge Regression method
Ridge_GS.best_params_


# In[87]:


plt.figure(figsize=(10,10))
plt.scatter(y_test, ridgereg_prediction)
plt.yscale('log')
plt.xscale('log')

p1 = max(max(ridgereg_prediction), max(y_test))
p2 = min(min(ridgereg_prediction), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize = 12)
plt.ylabel('Predictions', fontsize = 12)
plt.axis('equal')
plt.show


# In[ ]:




