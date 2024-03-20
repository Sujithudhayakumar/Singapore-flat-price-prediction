SINGAPORE FLAT RESALE PRICE PREDICTION WITH MACHINE LEARNING –          LINEAR REGRESSON
PROBLEM STATEMENT:
     The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.
TOOLS USED:
•	PYTHON - GOOGLE COLAB                                                                                                                                                                                                                                                                                 
•	MACHINE LEARNING    
•	POWER POINT
•	STREAMLIT    
•	GITHUB

IMPORT LIBARARIES:
   import libraries for handle the given data and handle with Machine learning algorithm (linear regression) for prediction of resale price of  Singapore flats

import pandas as pd
import streamlit as st
import streamlit_option_menu
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
DATA SOURCE:
   we have a 4csv files each representing the specific time. The time periods are 1990 -1999 ,2000-2012 ,2012-2014, 2015-2016, 2017 onwards. We want to setup these datasets as a one data frame
data = pd.read_csv('/content/ResaleFlatPricesBasedonApprovalDate19901999 (2).csv')
data1= pd.read_csv('/content/ResaleFlatPricesBasedonApprovalDate2000Feb2012 (1).csv')
data2 = pd.read_csv('/content/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016 (1).csv')
data3=pd.read_csv('/content/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014 (1).csv')
data4=pd.read_csv('/content/ResaleflatpricesbasedonregistrationdatefromJan2017onwards (1).csv')

dataframe=(data,data1,data2,data3,data4)
dataset=pd.concat(dataframe)

source link:
DATA PREPROCESSING:
   After merge all these  data we want to clean our data for better accuracy while predicting the resale price.
   Data pre processing:
o	drop null values
o	find and remove outliers of the column
o	encoding the data
 after that take a needed columns from the dataset for the prediction

LINEAR REGRESSION:
    To increase the accuracy of the prediction first normalise the data using LinearRegression() . Setup the new data as a trained 80% and tested data 20%. Here the dependent variable(y). While other features are independent variable

  Now it’s time to check the Accuracy of our machine learning model .I have check various evaluation metrics those are mean_squared_error(MSE), mean_absolute_error(MAE), r2_squared score of 0.5465 it means the accuracy of the model is 54.5%

CONCLUTION:
     In conclusion, predicting flat prices is a complex task that involves analyzing numerous variables such as location, property features, economic indicators, and market trends. While various methods, including statistical models, machine learning algorithms, and expert analysis, can be employed for price prediction, it's essential to acknowledge the inherent uncertainty and limitations in any forecasting endeavor.
