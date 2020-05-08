#!/usr/bin/env python
# coding: utf-8

# # This is Linear Regression Model to predict solar wind. 
# ## This was part of my Capastone project for CSE 485-486
# ### Project Name: Predicting Solar Wind Conditions with Machine Learning – Team Helios
# https://psyche.asu.edu/get-involved/capstone-projects/predicting-solar-wind-conditions-with-machine-learning-team-helios/
# ### James Niroomand, Arizona State University, Spring 2020
# 

# ## Project Overview
# We took the OMNI and ARTEMIS Spacecraft mission dataset to build a machine learning model
# The purpose of the model is to predict ARTEMIS Ion density, given OMNI Ion Density and other features such as Omni Latitude, Longitude, magnitude average and date/time. 
# Also, we trained our model with the difference in latitude and longitude between OMNI and ARTEMIS dataset. 
# 
# ### Datasets
# Dataset: from NASA  https://spdf.gsfc.nasa.gov/data_orbits.html 
# 
# OMNI and ARTEMIS dataset is more complete and clean compare to other datasets that are available and this was one of the reasons we chose these datasets. 
#     1. First I took the combined hourly dataset from March to October of 2017 and March to October of 2018. This is a large dataset with 3172 observations.It is a great way to start building and training our model. 
#     2. After I completed the training of my model with a large dataset, I focused on taking a smaller sample. Therefore, I took the dataset for May and June of 2018 and trained my model. 
# 
# ### Machine Learning Algorithm
# For this model we will be using Linear Regression model which we import from sklearn Library

# In[134]:


# libraries to import 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
from matplotlib.pyplot import figure
import seaborn as seabornInstance 
import datetime as dt
from sklearn.tree import DecisionTreeRegressor


# ###  Hourly Data from March 2017 to October 2017 and March 2018 to October 2018 from OMNI and ARTEMIS Spacecraft missions

# In[135]:


# Dataset from 2017 and 2018 
combinedDataFrom2017and2018 = pd.read_csv("../Artemis1and2YearRedux.csv")


# In[136]:


count_row = combinedDataFrom2017and2018.shape[0] 
print(count_row)


# In[137]:


# the start of our dataset
combinedDataFrom2017and2018.head()


# In[138]:


# end of our dataset 
combinedDataFrom2017and2018.tail()


# In[139]:


# here I am deleting some variables which i do not need for this model.
del combinedDataFrom2017and2018['Time_offset_hours']
del combinedDataFrom2017and2018['new_time']
del combinedDataFrom2017and2018['EPOCH_TIME__yyyy-mm-ddThh:mm:ss.sssZ']
del combinedDataFrom2017and2018['ARTEMIS_DIST_AU']
del combinedDataFrom2017and2018['ARTEMIS_LAT_DEG']
del combinedDataFrom2017and2018['ARTEMIS_LONG_DEG']
del combinedDataFrom2017and2018['SCALED_ARTEMIS_DENSITY']
del combinedDataFrom2017and2018['SCALED_ARTEMIS_MAG_AVG']


# In[140]:


combinedDataFrom2017and2018.head()


# In[141]:


# renaming the dataset
# here the latittude and longitude are the differences between the lat and long of Omni and Artemis
combinedDataFrom2017and2018.columns = [ "Date/Time",'Omni Latitude', 'Omni Longitude', "Omni Mag Average", "Omni speed", 'Omni Ion Density','Artemis Mag Average', 'Artemis Ion Densitity', 'Artemis Speed', 'Latitude Differences', 'Longitude Differences']


# In[142]:


# setting the data and time to represent hours and removing extra stuff
combinedDataFrom2017and2018['Date/Time'] = combinedDataFrom2017and2018['Date/Time'].values.astype('<M8[h]')


# #### Pulling the date and time and assigning it to its column

# In[143]:


combinedDataFrom2017and2018['Year'] = combinedDataFrom2017and2018['Date/Time'].dt.year
combinedDataFrom2017and2018['Month'] = combinedDataFrom2017and2018['Date/Time'].dt.month
combinedDataFrom2017and2018['Day'] = combinedDataFrom2017and2018['Date/Time'].dt.day
combinedDataFrom2017and2018['Hour'] = combinedDataFrom2017and2018['Date/Time'].dt.hour


# Here we are taking our ranamed and cleaned data set and storing it into CSV

# In[144]:


combinedDataFrom2017and2018.to_csv('cleanedCombinedData.csv', index = False)
combinedDataFrom2017and2018.head()


# ### We only need to pull certain features for our model to train on
#     1.We will only pull OMNI and OMNI Speed columns and make a Linear Regression Model based on these two variables and see how good our model is doing 
#     2. Next, I will pull OMNI Speed, Latitude, and Longitude differences between Omni and ARTEMIS Spacecraft, Omni Ion Density, and ARTEMIS Ion Density. We will be using these variables to build a multiple Linear Regression model to predict ARTEMIS Ion Density.
#     4. Furthermore, we will focus on training our model with OMNI latitude and longitude instead of their differences. 
#     5. Finally, we will take Omni Speed, Omni Mag Average, Omni Ion Density, Latitude, Longitude along with Year, Month, Day and hour to predict Artemis Ion Density
#     

# In[49]:


# Omni Speed and Artemis Speed in Km
Omni_Speed_and_Artemis_Speed = pd.read_csv("cleanedCombinedData.csv", usecols = ['Omni speed','Artemis Speed'])

# Pulling Omni Latitude, Longitude, Speed and Ion Density to predict Artemis Ion Density
Omni_long_lat_speed_ion_to_predict_Artemis_Ion_Density = pd.read_csv("cleanedCombinedData.csv",usecols = ['Omni Latitude','Omni Longitude','Omni speed','Omni Ion Density', 'Artemis Ion Densitity'])
# Omni Features to predict Artemis Ion Density
Omni_Features_with_Artemis_Ion_Density = pd.read_csv("cleanedCombinedData.csv",usecols = ['Omni speed','Omni Ion Density', 'Artemis Ion Densitity','Latitude Differences','Longitude Differences'])
# Omni Features with dates to predict Artemis Ion Density
Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density = pd.read_csv("MayandJune2018.csv",usecols = ['Omni Mag Average','Omni speed','Omni Ion Density', 'Artemis Ion Densitity','Latitude Differences','Longitude Differences','Year','Month','Day','Hour'])

_Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density_ = pd.read_csv("MayandJune2018.csv",usecols = ['Omni Latitude','Omni Longitude','Omni Mag Average','Omni speed','Omni Ion Density','Artemis Ion Densitity','Year','Month','Day','Hour'])



# In[145]:


Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density.head()


# #### Ploting the relationship between Omni and Artemis Speed

# In[15]:


# ploting the ion density of Omni dataset vs Ion Density Artemis
plt.plot(Omni_Speed_and_Artemis_Speed['Omni speed'], Omni_Speed_and_Artemis_Speed['Artemis Speed'], 'o', color = 'cadetblue', label = 'test')
plt.title('Omni Speed vs Artemis Speed')
plt.xlabel('Omni Speed')
plt.ylabel('Artemis Speed')
plt.show()


# #### Ploting the relationship between Omni and Artemis Ion Density

# In[146]:


plt.plot(Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density['Omni Ion Density'], Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density['Artemis Ion Densitity'], 'o', color = 'cadetblue', label = 'test')
plt.title('Omni Ion Densitity vs Artemis Ion Densitity')
plt.xlabel('Omni Ion Densitity')
plt.ylabel('Artemis Ion Densitity')
plt.show()


# # Measuring the Correlation in our data

# In[17]:


Omni_Speed_and_Artemis_Speed.corr()


# In[147]:


# finding relationship in our data
Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density.corr()


# - Very Strong relationship(|r| >0.8)
# - Strong Relationship (0.6 <= |r|)
# - Moderate Relatuin
# - Weak Relationship (|r| >= 0.2)
# - Very weak relationship (|r|)

# # Creating a Statistical Summary 

# In[148]:


Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density.describe()


# In[20]:


Omni_Speed_and_Artemis_Speed.describe()


# # Building our Model

# In[149]:


Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density.tail()


# In[152]:


# Spliting the data
X = Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density[['Omni Mag Average','Omni speed','Omni Ion Density','Latitude Differences','Longitude Differences','Year','Month','Day','Hour']].values
# Y is our output
Y = Omni_long_lat_speed_ion_mag_to_predict_Artemis_Ion_Density['Artemis Ion Densitity'].values
# X = Omni_Speed_and_Artemis_Speed[['Omni speed']].values
# Y = Omni_Speed_and_Artemis_Speed[['Artemis Speed']].values

# X = Omni_Features_with_Artemis_Ion_Density[['Omni speed','Omni Ion Density','Latitude Differences','Longitude Differences' ]].values
# Y = Omni_Features_with_Artemis_Ion_Density[['Artemis Ion Densitity']].values


# In[153]:


# spliting the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)


# # Creating and Fitting the Model
# Linear Regression Equation 
# 
# 
# #### Y = B0 + B1*X1 + B2*X2 + e
# 
# 
# The variables in the model are:
# 
# 1. Y, the response variable;
# 2. X1, the first predictor variable;
# 3. X2, the second predictor variable; and
# 4. e, the residual error, which is an unmeasured variable.
# 
# The parameters in the model are:
# 
# 1. B0, the Y-intercept;
# 2. B1, the first regression coefficient;
# 3. B2, the second regression coefficient.

# In[154]:


regr = LinearRegression()  
regr.fit(X_train, y_train)


# In[155]:


print(len(X_test))


# In[161]:


# Here I am pulling the year and month from our t
month = []
year = []
hour = []
day = []

# for i in range(200,250):
#     month.append(X_test[i][6])
    
# for i in range(200,250):
#     day.append(X_test[i][7])
       
# for i in range(200,250):
#     hour.append(X_test[i][8]) 
    
# for i in range(len(X_test)):
#     year.append(X_test[i][5])
index = []

for i in range(52):
    index.append(str(X_test[i][5]) + ',' + str(X_test[i][6]) + ',' + str(X_test[i][7])+ ',' + str(X_test[i][8]) + ':00')

# middle = endDate = str(X_test[25][5]) + ',' + str(X_test[25][6]) + ',' + str(X_test[25][7])+ ',' + str(X_test[25][8])

# endDate = str(X_test[52][5]) + ',' + str(X_test[52][6]) + ',' + str(X_test[52][7])+ ',' + str(X_test[52][8])

# print(startDate)
# print(middle)



# print(endDate)

print(index)


# In[162]:


# indexDate = "2018, 05, 18, 2:00 AM"
# endIndex = "2018, 05, 23, 10:00 AM"
index  = [s.replace('.0', '') for s in index] # remove all the 8s 


# In[158]:


# making prediction with our test data 
# We trained our model with 80 percent of our sample size and here we are making a prediction 
# with 20 percent of the sample size 
y_pred = regr.predict(X_test)
predictedData = pd.DataFrame({'Actual Ion Density': y_test.flatten(), 'Predicted Ion Density': y_pred.flatten()})
predictedData[:10]

# predictedData = pd.DataFrame({'Actual Speed (km)': y_test.flatten(), 'Predicted Speed (km)': y_pred.flatten()})
# predictedData[:10]


# In[159]:


#To retrieve the intercept:
print("Intercept %.2f" % regr.intercept_)
#For retrieving the slope:
print("slope: ",  regr.coef_)


# In[160]:


# getting the prediction of the model

from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance and Accuracy of our model is: %.2f" % regr.score(X_test, y_test))


# # Plotting

# In[170]:


get_ipython().run_line_magic('matplotlib', 'inline')
# X_test is the test data set.

size=20
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)

plt.xticks( range(52), index)
plt.xticks(rotation=270)
plt.title("Predicting Artemis Ion Density")
plt.plot(y_pred, 'm', label='Predicted Data of ARTEMIS Ion Density with 0.81 accuracy ')
plt.plot(y_test, label='Actual Data is hourly data from May 2018 ')
plt.ylabel("Ion Density (nT)")
plt.xlabel("Hourly Data from May and June 2018" )

plt.legend()
#plt.xlim([0,53])
#plt.xlim([400,450])
plt.show()


# # Evaluating our Model
# ## Using the Statsmodel
# 

# In[171]:



X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 


# Confidence Interval
# 

# In[172]:


model.conf_int()


# # Hypothesis Testing 
#        - Null Hypothesis: There is no relationship between our data
#        - Alternative Hypothesis: There is relationship between our data

# In[173]:


model.pvalues


# The p-value represents the probability coefficent equals to 0. We want a p-value that is less than 0.05, if it is we can reject the null hypothesis. 

# # Next we can evaluate how well our model is doing
# - Mean Absolute Error(MAE): Is the mean of the absolute value of the errors. This metric gives an idea of magnitude but no idea about direction 
# - Mean Squared Error(MSE): Is the mean of the squared errors.
# - Root Mean Squared Error(RMSE): Is the square root of the mean of the squared errors. 

# In[174]:


# Mean Squared Error 
model_mse = mean_squared_error(y_test, y_pred)
# Mean Absolute Error
model_mae = mean_absolute_error(y_test, y_pred)
# Rppt mean Squared Error
model_rmse = math.sqrt(model_mse)

print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))


# # R-Squared 

# - The R-Squared metric provides us a way to measure the goodness of fit or how well our data fits the model. The highest the R-squared metric the better the data fit our model. 

# In[175]:


r2 = r2_score(y_test,y_pred)
print("R2 Score {:.2}".format(r2))


# # Create a summar of the model output

# In[2093]:


print(model.summary())


# Once you have obtained your error metric/s, take note of which X’s have minimal impacts on y. Removing some of these features may result in an increased accuracy of your model.

# In[1512]:


# we save our model 
import pickle
# save the model
with open("Linear_Regression_Model", 'wb') as f:
    pickle.dump(regr,f)


# In[1513]:


# this is how we open our model for fututre use.
with open('Linear_Regression_Model', 'rb') as pickle_file:
    reg_mode_2 = pickle.load(pickle_file)


# In[ ]:




