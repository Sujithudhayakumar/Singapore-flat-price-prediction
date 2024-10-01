import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the title of the app
st.title('Resale Flat Price Prediction')


# Input fields for user to enter details
st.header('Enter the details of the flat')

floor_area_sqm = st.number_input('Floor Area (in sqm)', min_value=20, max_value=200, value=100)
flat_type = st.selectbox('Flat Type', options=['1 Room', '2 Room', '3 Room', '4 Room', '5 Room', 'Executive', 'Multi-Generation'])
flat_model = st.selectbox('Flat Model', options=['Model A', 'Model B', 'Model C', 'Model D'])
storey_range = st.selectbox('Storey Range', options=['1 TO 3', '4 TO 6', '7 TO 9', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'])
lease_commence_date = st.number_input('Lease Commence Date', min_value=1900, max_value=2024, value=2000)

# Calculate the remaining lease years
lease_remaining_year = 99 - (2024 - lease_commence_date)

# Encode the flat_type and flat_model as numeric values using the same encoder
flat_type_encoder = LabelEncoder()
flat_model_encoder = LabelEncoder()

flat_type_encoded = flat_type_encoder.fit_transform([flat_type])[0]
flat_model_encoded = flat_model_encoder.fit_transform([flat_model])[0]

# Compute the median of the storey range
def get_median(x):
    split_list = x.split('TO')
    float_list = [float(i) for i in split_list]
    median = np.median(float_list)
    return median

storey_median = get_median(storey_range)

# Create a dataframe for the input data
input_data = pd.DataFrame({
    'floor_area_sqm': [floor_area_sqm],
    'flat_type': [flat_type_encoded],
    'flat_model': [flat_model_encoded],
    'storey_median': [storey_median],
    'lease_remaining_year': [lease_remaining_year]
})

# Predict the resale price using the model
if st.button('Predict Resale Price'):
    prediction = model.predict(input_data)[0]
    st.subheader(f'Estimated Resale Price: ${prediction:,.2f}')

# Display the model performance metrics
st.sidebar.header('Model Performance')
mse = 24120125.57  # Replace with your actual MSE
mae = 3452.78  # Replace with your actual MAE
r2 = 0.82  # Replace with your actual R2 score
st.sidebar.write(f'Mean Squared Error: {mse:,.2f}')
st.sidebar.write(f'Mean Absolute Error: {mae:,.2f}')
st.sidebar.write(f'R-Squared: {r2:.2f}')
