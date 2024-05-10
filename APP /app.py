import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Setting page configuration
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# Reading the data on Lat and Long of all the MRT Stations in Singapore
data_files = [
    '/content/ResaleFlatPricesBasedonApprovalDate19901999 (2).csv',
    '/content/ResaleFlatPricesBasedonApprovalDate2000Feb2012 (1).csv',
    '/content/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016 (1).csv',
    '/content/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014 (1).csv',
    '/content/ResaleflatpricesbasedonregistrationdatefromJan2017onwards (1).csv'
]

dataset = pd.concat([pd.read_csv(file) for file in data_files])

# Data preprocessing
dataset['flat_type'] = dataset['flat_type'].astype(str)
dataset['flat_model'] = dataset['flat_model'].astype(str)
dataset["resale_price"] = dataset['resale_price'].astype("float")
dataset['floor_area_sqm'] = dataset['floor_area_sqm'].astype('float')
dataset['lease_commence_date'] = dataset['lease_commence_date'].astype('int64')
dataset['lease_remaining_year'] = 99 - (2024 - dataset['lease_commence_date'])
dataset['flat_type'] = LabelEncoder().fit_transform(dataset['flat_type'])
dataset['flat_model'] = LabelEncoder().fit_transform(dataset['flat_model'])

# Function to remove outliers
def remove_outliers(df, column, multiplier=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * multiplier)
    upper_bound = q3 + (iqr * multiplier)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Removing outliers
dataset = remove_outliers(dataset, 'flat_type')

# Sidebar navigation
with st.sidebar:
    selected = st.radio("Main Menu", ["About Project", "Predictions"])

# About Project Section
if selected == "About Project":
    st.markdown("# Singapore Resale Flat Prices Prediction")
    st.markdown("### Technologies: Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### Overview: This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. Resale prices are influenced by a wide variety "
                "of criteria, including location, type of apartment, total square footage, and lease length. "
                "The model assists buyers and sellers in evaluating the worth of a flat after resale.")

# Predictions Section
if selected == "Predictions":
    st.markdown("# Predicting Results based on Trained Models")
    st.markdown("### Predicting Resale Price (Regression Task)")

    # New Data inputs from the user for predicting the resale price
    with st.form("form1"):
        floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=10.0, max_value=500.0)
        flat_type = st.selectbox("Flat Type", dataset['flat_type'].unique())
        flat_model = st.selectbox("Flat Model", dataset['flat_model'].unique())
        storey_median = st.number_input('Storey Median', min_value=1, max_value=99)
        lease_remaining_year = st.number_input('Lease Remaining Year', min_value=0, max_value=99)

        # Submit Button for PREDICT RESALE PRICE
        submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

        if submit_button:
            X = dataset[['floor_area_sqm', 'flat_type', 'flat_model', 'lease_remaining_year']]
            y = dataset['resale_price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and fit the model
            lm = LinearRegression()
            lm.fit(X_train, y_train)

            # Sending the user-entered values for prediction to our model
            try:
                new_pred = lm.predict([[floor_area_sqm, flat_type, flat_model, lease_remaining_year]])
                st.write('## Predicted resale price:', new_pred[0])

            except Exception as e:
                st.error("Error occurred while predicting. Please ensure all fields are filled correctly.")
