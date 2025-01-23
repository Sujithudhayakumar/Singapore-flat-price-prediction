#IMPORT LIBRARIES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import statistics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#LOAD THE GIVEN DATA
data = pd.read_csv(r'C:\Users\LENOVO\Documents\dataset\ResaleFlatPricesBasedonApprovalDate2000Feb2012 (1).csv')
data1= pd.read_csv(r'C:\Users\LENOVO\Documents\dataset\ResaleFlatPricesBasedonApprovalDate19901999 (2).csv')
data2 = pd.read_csv(r'C:\Users\LENOVO\Documents\dataset\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016 (1).csv')
data3=pd.read_csv(r'C:\Users\LENOVO\Documents\dataset\ResaleflatpricesbasedonregistrationdatefromJan2017onwards (1).csv')
data4=pd.read_csv(r'C:\Users\LENOVO\Documents\dataset\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014 (1).csv')

# #MERGE THE ALL DATA WITH CONCAT
dataframe=(data,data1,data2,data3,data4)
dataset=pd.concat(dataframe)

#CHECKING THE NULL VALUES AND REMOVE IT
dataset.isnull().sum()
dataset.dropna(inplace=True)

#TYPE COSTING THE NEEDED COLUMNS
dataset['flat_type']=dataset['flat_type'].astype(str)
dataset['flat_model']=dataset['flat_model'].astype(str)
dataset["resale_price"]=dataset['resale_price'].astype("float")
dataset['floor_area_sqm'] = dataset['floor_area_sqm'].astype('float')
dataset['lease_commence_date'] = dataset['lease_commence_date'].astype('int')
dataset['lease_remaining_year'] = 99 - (2024 - dataset['lease_commence_date'])
dataset['flat_type']=dataset['flat_type'].astype(str)

#ENCODE THE COLUMN AS NUMERIC VALUE
town = dataset['flat_model']
label_encoder = LabelEncoder()
dataset['flat_model']= label_encoder.fit_transform(town)

type_ = dataset['flat_type']
dataset['flat_type']= label_encoder.fit_transform(type_)

#FROM THE STOREY_RANGE COLUMN FIND THE MEDIAN AND SAVE THE COLUMN AS STOREY_MEDIAN
def get_median(x):
  split_list= x.split('TO')
  float_list= [float(i) for i in split_list]
  median= statistics.median(float_list)
  return median
dataset['storey_median']= dataset['storey_range'].apply(lambda x:get_median(x))

#USING THE BOXPLOT FOR CHECKING THE OUTLIERS
column=['storey_median','lease_remaining_year','flat_model','flat_type','floor_area_sqm',"resale_price"]
for i in column:
  plt.figure(figsize=(8,6))
  sns.boxplot(data=dataset,x=i)
  plt.title(f'box plot {i}')
  plt.xlabel(i)
  plt.show()

#REMOVE THE OUTLIERS FROM THE ALL COLUMNS
def remove_outliers(df,column,multiplier=1.5):
  q1=df[column].quantile(0.25)
  q3=df[column].quantile(0.75)
  iqr=q3-q1
  lowwer_bound=q1-(iqr*multiplier)
  upper_bound=q3+(iqr*multiplier)
  df_cleaned=df[(df[column]>=lowwer_bound)&(df[column]<=upper_bound)]
  return df_cleaned
df_cleaned=remove_outliers(dataset,'flat_type')

#DEFINE X AND y AND SPLIT THE DATA AS TRAIN AND TEST
X =df_cleaned[['floor_area_sqm','flat_type','flat_model','storey_median','lease_remaining_year']]
y =df_cleaned['resale_price']

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize and fit the model
model=RandomForestRegressor()
model.fit(X_train,y_train)

#PREDICTION USING THE REGRESSION MODEL
prediction=model.predict(X_test)
with open('Random_Forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


#Model evaluation
mse =mean_squared_error(y_test,prediction)
mae=mean_absolute_error(y_test,prediction)
r2=r2_score(y_test,prediction)
print('mean_squared_error:',mse)
print('mean_absolute_error:',mae)
print('r2_score:',r2)