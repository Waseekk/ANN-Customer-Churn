import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load resources with error handling
try:
    model = tf.keras.models.load_model('churn_model.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
    onehot_encoder_geography = pickle.load(open('one_hot_geography.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Streamlit app
st.title('Advanced Customer Churn Prediction App')
st.markdown(
    """
    This app predicts whether a customer will churn based on their profile and transaction details. 
    Please fill in the required fields to get a prediction.
    """
)

# User Input Section
st.header("Customer Profile")
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0], help="Select the customer's region")
    gender = st.selectbox('Gender', label_encoder_gender.classes_, help="Select the customer's gender")
    age = st.slider('Age', 18, 100, help="Select the customer's age")
    credit_score = st.number_input('Credit Score (0-1000)', min_value=0, max_value=1000, step=1)

with col2:
    balance = st.number_input('Balance', min_value=0.0, step=1000.0, help="Enter the customer's account balance")
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0, help="Enter the customer's estimated salary")
    tenure = st.slider('Tenure (years)', 0, 10, help="Select the tenure with the bank")
    num_of_products = st.slider('Number of Products', 1, 4, help="Select the number of products the customer has")

st.header("Additional Information")
col3, col4 = st.columns(2)

with col3:
    has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'], help="Does the customer have a credit card?")
with col4:
    is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'], help="Is the customer an active member?")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary],
})

geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
st.header("Prediction")
with st.spinner('Predicting...'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

# Display Results
if prediction_proba > 0.5:
    st.success('The customer is likely to churn.')
else:
    st.success('The customer is unlikely to churn.')

# Show prediction probability
st.write(f"Prediction Probability: {prediction_proba:.2%}")
fig, ax = plt.subplots()
sns.barplot(x=['Not Churn', 'Churn'], y=[1 - prediction_proba, prediction_proba], palette="Blues", ax=ax)
ax.set_title("Churn Prediction Probability")
st.pyplot(fig)

# Input Summary
st.header("Input Summary")
st.write(input_data)
