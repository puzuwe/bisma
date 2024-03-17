pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle

# Function to preprocess test data
def preprocess_test_data(df_test):
    # Converting binary categorical variables to numerical
    binary_mapping = {'yes': 1, 'no': 0}
    binary_columns = ['surgery', 'surgical_lesion', 'cp_data']  # Remove 'age' from binary columns
    
    for col in binary_columns:
        df_test[col] = df_test[col].replace(binary_mapping)

    # Convert 'age' column to binary (1 for 'adult', 0 for 'young')
    df_test['age'] = df_test['age'].map({'adult': 1, 'young': 0})
    # One-hot encoding for categorical variables
    categorical_columns = ['temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time',
                           'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux',
                           'rectal_exam_feces', 'abdomen', 'abdomo_appearance']
    
    df_test = pd.get_dummies(df_test, columns=categorical_columns)

    # Drop 'id' column
    X_test_id = df_test['id']
    X_test = df_test.drop(columns='id')

    # Add missing columns with default values
    missing_columns = ['nasogastric_reflux_slight', 'pain_slight', 'peristalsis_distend_small', 'rectal_exam_feces_serosanguious']
    for col in missing_columns:
        X_test[col] = 0
    
    return X_test, X_test_id

# Load stacked model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('stacked_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Mapping dictionary for predicted results
prediction_mapping = {0: 'died', 1: 'euthanized', 2: 'lived'}

# App title
st.title('Data Prediction')

# File uploader for test data
st.sidebar.title('Upload Data')
# uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
data_option = st.sidebar.radio('Choose Data Input Option:', ('Upload CSV', 'Enter Data Manually'))
