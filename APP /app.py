
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
from sklearn.model_selection import GridSearchCV

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# -------------------------Reading the data on Lat and Long of all the MRT Stations in Singapore------------------------
data = pd.read_csv('/content/ResaleFlatPricesBasedonApprovalDate19901999 (2).csv')
data1= pd.read_csv('/content/ResaleFlatPricesBasedonApprovalDate2000Feb2012 (1).csv')
data2 = pd.read_csv('/content/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016 (1).csv')
data3=pd.read_csv('/content/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014 (1).csv')
data4=pd.read_csv('/content/ResaleflatpricesbasedonregistrationdatefromJan2017onwards (1).csv')

dataframe=(data,data1,data2,data3,data4)
dataset=pd.concat(dataframe)
dataset
#
dataset['flat_type']=dataset['flat_type'].astype(str)
dataset['flat_model']=dataset['flat_model'].astype(str)
dataset["resale_price"]=dataset['resale_price'].astype("float")
dataset['floor_area_sqm'] = dataset['floor_area_sqm'].astype('float')
dataset['lease_commence_date'] = dataset['lease_commence_date'].astype('int64')
dataset['lease_remaining_year'] = 99 - (2024 - dataset['lease_commence_date'])
dataset['flat_type']=dataset['flat_type'].astype(str)

#
from sklearn.preprocessing import LabelEncoder

town = dataset['flat_model']
labelencoder = LabelEncoder()
dataset['flat_model']= labelencoder.fit_transform(town)

type_ = dataset['flat_type']
labelencoder = LabelEncoder()
dataset['flat_type']= labelencoder.fit_transform(type_)

#
import statistics
def get_median(x):
  split_list= x.split('TO')
  float_list= [float(i) for i in split_list]
  median= statistics.median(float_list)
  return median
dataset['storey_median']= dataset['storey_range'].apply(lambda x:get_median(x))
dataset

#
import seaborn as sns
import matplotlib.pyplot as plt
column=['storey_median','lease_remaining_year','flat_model','flat_type','floor_area_sqm',"resale_price"]
for i in column:
  plt.figure(figsize=(8,6))
  sns.boxplot(data=dataset,x=i)
  plt.title(f'box plot {i}')
  plt.xlabel(i)
  plt.show()

  #
def remove_outliers(df,column,multiplier=1.5):
  q1=df[column].quantile(0.25)
  q3=df[column].quantile(0.75)
  iqr=q3-q1
  lowwer_bound=q1-(iqr*multiplier)
  upper_bound=q3+(iqr*multiplier)
  df_cleaned=df[(df[column]>=lowwer_bound)&(df[column]<=upper_bound)]
  return df_cleaned
df_cleaned=remove_outliers(dataset,'flat_type')






# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    with st.form("form1"):
        # -----New Data inputs from the user for predicting the resale price-----
        street_name = st.text_input("Street Name")
        block = st.text_input("Block Number")
        floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
        lease_commence_date = st.number_input('Lease Commence Date')
        storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

        # -----Submit Button for PREDICT RESALE PRICE-----
        submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

        lm = LinearRegression()

        if submit_button is not None:
            with open('linear_regression_model.pkl', 'wb') as model_file:
                pickle.dump(lm, model_file)

            X = df_cleaned[['floor_area_sqm', 'flat_type', 'flat_model', 'storey_median', 'lease_remaining_year']]
            y = df_cleaned['resale_price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and fit the model
            lm = LinearRegression()
            lm.fit(X_train, y_train)

            # -----Sending the user enter values for prediction to our model-----
            try:
                new_pred = lm.predict(X_test)
                st.write('## :green[Predicted resale price:] ', (new_pred)[0])

            except Exception as e:
                st.write("Enter the above values to get the predicted resale price of the flat")
