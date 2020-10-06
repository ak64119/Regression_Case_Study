# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:33:19 2019

@author: akshay
"""

#==============================================================================
# Predicting Price of Pre-owned Cars
#==============================================================================

#==============================================================================
# Import Packages
#==============================================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib qt

#==============================================================================
# Setting Dimensions of plots
#==============================================================================

sns.set(rc={'figure.figsize':(11.7,8.27)})

#==============================================================================
# Reading Data
#==============================================================================

cars_data = pd.read_csv("cars_sampled.csv")

#==============================================================================
# Creating Copy
#==============================================================================

cars = cars_data.copy()

#==============================================================================
# Exploring Structure of the Data
#==============================================================================

cars.info()

#==============================================================================
# Summarizing Data
#==============================================================================

cars.describe()
pd.set_option('display.float_format', lambda x: '%.3f'  % x)
cars.describe()

# Display maximum number of columns
pd.set_option('display.max_columns',500)
cars.describe()

#==============================================================================
# Dropping Unwanted Columns
#==============================================================================

col = ['name', 'dateCrawled', 'dateCreated', 'postalCode','lastSeen']
cars = cars.drop(columns = col, axis =1)

#==============================================================================
# Removing Duplicate Records
#==============================================================================

cars.drop_duplicates(keep = 'first', inplace = True)
#470 Duplicate Records dropped

#==============================================================================
# Data Cleaning
#==============================================================================

# No. of missing values in each column
cars.isnull().sum()

#Variable yearOfRegistration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration'] > 2018)
sum(cars['yearOfRegistration'] < 1950)
sns.regplot(x='yearOfRegistration',y='price', scatter = True,
            fit_reg = False, data=cars)
#Working Range- 1950 and 2018

#Variable Price
price_count = cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y = cars['price'])
sum(cars['price'] < 100)
sum(cars['price'] > 150000)
#Working Range- 100 and 150000

#Variable powerPS
power_count = cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y = cars['powerPS'])
sns.regplot(x='powerPS',y='price', scatter = True,
            fit_reg = False, data=cars)
sum(cars['powerPS'] < 10)
sum(cars['powerPS'] > 500)
#Working Range- 10 and 500
###Based on the reading regarding what amount of power
###is required to run a car.

#==============================================================================
# Working Range of Data
#==============================================================================

#Working Range of Data

cars = cars[
        (cars.yearOfRegistration <= 2018)
      & (cars.yearOfRegistration >= 1950)
      & (cars.price >= 100)
      & (cars.price <= 150000)
      & (cars.powerPS >= 10)
      & (cars.powerPS <= 500)]
# ~6700 records are dropped

# Further to simplify - variable reduction
# Combining yearOfRegistration and monthOfRegistration

cars['monthOfRegistration']/= 12

# Creating new variable Age by adding yearOfRegistration and monthOfRegistration
cars['Age'] = ((2018-cars['yearOfRegistration']) + cars['monthOfRegistration'])
cars['Age'] = round(cars['Age'],2)
cars.describe()

# Dropping yearOfRegistration and monthOfRegistration columns
cars = cars.drop(columns =['yearOfRegistration','monthOfRegistration'], axis =1)

# Visualizing Parameters

# Age
sns.distplot(cars['Age'])
sns.boxplot(y = cars['Age'])

# Price
sns.distplot(cars['price'])
sns.boxplot(y = cars['price'])

# powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y = cars['powerPS'])

# Visualizing parameters after narrowing working range
# Age vs Price
sns.regplot(x='Age',y='price', scatter = True,
            fit_reg = False, data=cars)

# powerPS vs Price
sns.regplot(x='powerPS',y='price', scatter = True,
            fit_reg = False, data=cars)

# Variable Seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns = 'count', normalize = True)
sns.countplot(x = 'seller', data = cars)
# Fewer Cars have 'commericial' => Insignificant

# Variable offerType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'], columns = 'count', normalize = True)
sns.countplot(x = 'offerType', data = cars)
# Fewer Cars have 'offer' => Insignificant

# Variable offerType
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'], columns = 'count', normalize = True)
sns.countplot(x = 'abtest', data = cars)
# Equally distributed
sns.boxplot(x='abtest',y='price', data=cars)
# For every price value there is almost 50-50 distribution
# Does not affect 'price' => Insignificant

# Variable vechileType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'], columns = 'count', normalize = True)
sns.countplot(x = 'vehicleType', data = cars)
sns.boxplot(x='vehicleType',y='price', data=cars)
# 'VehicleType' affects price => Significant

# Variable vechileType
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'], columns = 'count', normalize = True)
sns.countplot(x = 'gearbox', data = cars)
sns.boxplot(x='gearbox',y='price', data=cars)
# 'gearbox' affects price => Significant

# Variable model
cars['model'].value_counts()
pd.crosstab(cars['model'], columns = 'count', normalize = True)
sns.countplot(x = 'model', data = cars)
sns.boxplot(x='model',y='price', data=cars)
# 'model' affects price => Significant

# Variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'], columns = 'count', normalize = True)
sns.countplot(x = 'kilometer', data = cars)
sns.boxplot(x='kilometer',y='price', data=cars)
# 'kilometer' affects price => Significant

# Variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns = 'count', normalize = True)
sns.countplot(x = 'fuelType', data = cars)
sns.boxplot(x='fuelType',y='price', data=cars)
# 'fuelType' affects price => Significant

# Variable Brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns = 'count', normalize = True)
sns.countplot(x = 'brand', data = cars)
sns.boxplot(x='brand',y='price', data=cars)
# 'brand' affects price => Significant

# Variable notRepairedDamage
# yes -  Car is damaged but not rectified
# no - Car was damaged but was rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns = 'count', normalize = True)
sns.countplot(x = 'notRepairedDamage', data = cars)
sns.boxplot(x='notRepairedDamage',y='price', data=cars)
# As expected the cars that require the damage to be repaired
#fall under lower proce range

#==============================================================================
# Removing Insignificant variables
#==============================================================================

col_2 = ['seller', 'offerType', 'abtest']
cars = cars.drop(columns = col_2, axis =1)
cars_copy = cars.copy()

#==============================================================================
# Checking Correlation
#==============================================================================

cars_select1 = cars.select_dtypes(exclude=[object])
correlation = cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

#==============================================================================

#==============================================================================
# Model Building
#==============================================================================

"""
We are going to build a Linear Regression and Random Forest model
on two sets of data.
1. Data obtained by omitting rows with any missing values.
2. Data obtained by inputting missing values.
"""

#==============================================================================
# Omitting Missing Values
#==============================================================================

cars_omit = cars.dropna(axis=0)

# Converting categorical variables to dummy variables
cars_omit = pd.get_dummies(cars_omit,drop_first=True)

#==============================================================================
# Importing Necessary libraries
#==============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#==============================================================================
# Model Building with Omitted Data
#==============================================================================

# Separating input and output features
x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
y1 = cars_omit['price']

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y1, "2. After (Taking log)":np.log(y1)})
prices.hist()
# Taking the log of prices shows a normal distribution curve. We will take 
# price with a log value because the range of price is huge, and we are taking
# a log to avoid it.

#Transforming price as a logarithmic value
y1 = np.log(y1)

# Splitting data as test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, 
                                                    random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#==============================================================================
# Baseline Model FOR Omitted Data
#==============================================================================

"""
We are making a base model by using test data mean value.
This is to set a benchmark and to compare with our regression model.
"""

# find the meanfor test data value
base_pred = np.mean(y_test)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred,len(y_test))

# finding the RMSE value
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test,base_pred))

print(base_root_mean_square_error)

#==============================================================================
# LINEAR REGRESSION WITH Omitted Data
#==============================================================================

# Setting Intercept as as true
lgr = LinearRegression(fit_intercept = True)

# fitting model
model_lin1 = lgr.fit(X_train,y_train)

# Predicting model on test data
cars_predictions_lin1 = lgr.predict(X_test)

# Computing MSE and RMSE Value
lin_mse1 = mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

# R squared value
# Explains the variability in Y
r2_lin_test1 = model_lin1.score(X_test,y_test)
r2_lin_train1 = model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

# Regression diagnostics - Residual plot analysis
residuals1= y_test - cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1, scatter = True,
            fit_reg = False, data=cars)
residuals1.describe()
# From the residual analysis, we can see that the mean is '0.003', i.e., the 
# predicted and actual values are really very close.


#==============================================================================
# Random Forest with Omitted Data
#==============================================================================

# Model Parameters
rf = RandomForestRegressor(n_estimators=100, max_features = "auto",
                           max_depth = 100, min_samples_split=10,
                           min_samples_leaf=4, random_state=1)

# Model fit
model_rf1 = rf.fit(X_train,y_train)

#Predicting model on test set
cars_prediction_rf1 = rf.predict(X_test)

# Computing MSE and RMSE Value
rf_mse1 = mean_squared_error(y_test,cars_prediction_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
# Explains the variability in Y
r2_rf_test1 = model_rf1.score(X_test,y_test)
r2_rf_train1 = model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)


#==============================================================================
# MODEL BUILDING WITH IMPUTED DATA
#==============================================================================

cars_imputed = cars.apply(lambda x:x.fillna(x.mean()) 
                            if x.dtype == 'float' else
                             x.fillna(x.value_counts().index[0]))

cars_imputed.isnull().sum()

# Convert categorical variables to dummy variables
cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

#==============================================================================
# Model Building with IMPUTED Data
#==============================================================================

# Separating input and output features
x2 = cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y2, "2. After (Taking log)":np.log(y2)})
prices.hist()
# Taking the log of prices shows a normal distribution curve. We will take 
# price with a log value because the range of price is huge, and we are taking
# a log to avoid it.

#Transforming price as a logarithmic value
y2 = np.log(y2)

# Splitting data as test and train
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, 
                                                    random_state=3)
print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)

#==============================================================================
# Baseline Model FOR IMPUTED Data
#==============================================================================

"""
We are making a base model by using test data mean value.
This is to set a benchmark and to compare with our regression model.
"""

# find the meanfor test data value
base_pred2 = np.mean(y_test2)
print(base_pred2)

# Repeating same value till length of test data
base_pred2 = np.repeat(base_pred2,len(y_test2))

# finding the RMSE value
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test2,base_pred2))

print(base_root_mean_square_error_imputed)

#==============================================================================
# LINEAR REGRESSION WITH IMPUTED Data
#==============================================================================

# Setting Intercept as as true
lgr2 = LinearRegression(fit_intercept = True)

# fitting model
model_lin2 = lgr2.fit(X_train2,y_train2)

# Predicting model on test data
cars_predictions_lin2 = lgr2.predict(X_test2)

# Computing MSE and RMSE Value
lin_mse2 = mean_squared_error(y_test2,cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

# R squared value
# Explains the variability in Y
r2_lin_test2 = model_lin1.score(X_test2,y_test2)
r2_lin_train2 = model_lin1.score(X_train2,y_train2)
print(r2_lin_test2,r2_lin_train2)

# Regression diagnostics - Residual plot analysis
residuals2= y_test2 - cars_predictions_lin2
sns.regplot(x=cars_predictions_lin2,y=residuals2, scatter = True,
            fit_reg = False, data=cars)
residuals2.describe()
# From the residual analysis, we can see that the mean is '0.003', i.e., the 
# predicted and actual values are really very close.

#==============================================================================
# Random Forest with Omitted Data
#==============================================================================

# Model Parameters
rf2 = RandomForestRegressor(n_estimators=100, max_features = "auto",
                           max_depth = 100, min_samples_split=10,
                           min_samples_leaf=4, random_state=1)

# Model fit
model_rf2 = rf.fit(X_train2,y_train2)

#Predicting model on test set
cars_prediction_rf2 = rf.predict(X_test2)

# Computing MSE and RMSE Value
rf_mse2 = mean_squared_error(y_test2,cars_prediction_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)

# R squared value
# Explains the variability in Y
r2_rf_test2 = model_rf1.score(X_test2,y_test2)
r2_rf_train2 = model_rf1.score(X_train2,y_train2)
print(r2_rf_test2,r2_rf_train2)

#==============================================================================
# Final Output
#==============================================================================
print("\n\n")
print("Metrics for model built from data where missing values were omitted")
print("R squared value for train from Linear Regression = %s"% r2_lin_train1)
print("R squared value for test from Linear Regression = %s"% r2_lin_test1)
print("R squared value for train from Random Forest = %s"% r2_rf_train1)
print("R squared value for test from Random Forest = %s"% r2_rf_test1)
print("Base RMSE of model built from data where missing values were omitted = %s"% base_root_mean_square_error)
print("RMSE value for test from Linear Regression = %s"% lin_rmse1)
print("RMSE value for test from Random Forest = %s"% rf_rmse1)
print("\n\n")
print("Metrics for model built from data where missing values were imputted")
print("R squared value for train from Linear Regression = %s"% r2_lin_train2)
print("R squared value for test from Linear Regression = %s"% r2_lin_test2)
print("R squared value for train from Random Forest = %s"% r2_rf_train2)
print("R squared value for test from Random Forest = %s"% r2_rf_test2)
print("Base RMSE of model built from data where missing values were omitted = %s"% base_root_mean_square_error_imputed)
print("RMSE value for test from Linear Regression = %s"% lin_rmse2)
print("RMSE value for test from Random Forest = %s"% rf_rmse2)
print("\n\n")

#==============================================================================
# End of Script
#==============================================================================