import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load the model
model = tf.keras.models.load_model('churn_model.h5')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the encoder
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
onehot_encoder_geography = pickle.load(open('one_hot_geography.pkl', 'rb'))

## streamlit app
st.title('Churn Churn Prediction App')

# user input
#geography=st.selectbox('Geography', ['France', 'Germany', 'Spain'])
geography=st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', 18, 100)
balance=st.number_input('Balance',step=1000)
credit_score=st.number_input('Credit Score',min_value=0, max_value=10, step=1)
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure')
num_of_products=st.slider('Number of Products')
has_cr_card=st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member=st.selectbox('Is Active Member', ['Yes', 'No'])


## prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card=='Yes' else 0],
    'IsActiveMember': [1 if is_active_member=='Yes' else 0],
    'EstimatedSalary': [estimated_salary],
})

geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
#geo_df=pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.categories_[0])
#gep_df=pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names(['Geography']))
geo_df=pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))     

input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

## scale the input data
input_data_scaled = scaler.transform(input_data)

## predict
prediction = model.predict(input_data_scaled)
#prediction_proba = model.predict_proba(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write('the cutomer will churn')
else:
    st.write('the cutomer will not churn')