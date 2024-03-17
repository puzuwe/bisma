
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

if data_option == 'Upload CSV':
    uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    if uploaded_file is not None:
       # Read the uploaded file
       df_test = pd.read_csv(uploaded_file)

       # Display the uploaded data
       st.sidebar.subheader('Uploaded Data:')
       st.sidebar.write(df_test)


       # Preprocess the test data
       X_test_preprocessed, X_test_id = preprocess_test_data(df_test)

       # Display preprocessed data
       st.subheader('Preprocessed  Data:')
       st.write(X_test_preprocessed)
       # Load the model
       model = load_model()

       # Make predictions
       predictions = model.predict(X_test_preprocessed)

       # Map predictions to labels
       predictions_labels = [prediction_mapping[prediction] for prediction in predictions]

       # Display predictions
       st.subheader('Predictions:')
       st.write(predictions_labels)
       # Display the shape of the predictions
       st.write(f"Predictions Shape: {len(predictions_labels)}")
else:
    st.info('Manual Entry.')
    surgery = st.sidebar.selectbox('Surgery', ['yes', 'no'])
    age = st.sidebar.selectbox('Age', ['adult', 'young'])
    rectal_temp = st.sidebar.number_input('Rectal Temperature', value=38.0)
    pulse = st.sidebar.number_input('Pulse', value=72)
    respiratory_rate = st.sidebar.number_input('Respiratory Rate', value=30)
    temp_of_extremities = st.sidebar.selectbox('Temperature of Extremities', ['cool', 'cold', 'warm', 'normal'])
    peripheral_pulse = st.sidebar.selectbox('Peripheral Pulse', ['reduced', 'normal', 'increased'])
    mucous_membrane = st.sidebar.selectbox('Mucous Membrane', ['pale_pink', 'pale_cyanotic', 'bright_pink', 'bright_red', 'dark_cyanotic'])
    capillary_refill_time = st.sidebar.selectbox('Capillary Refill Time', ['less_3_sec', 'more_3_sec'])
    pain = st.sidebar.selectbox('Pain', ['depressed', 'mild_pain', 'moderate', 'severe_pain', 'extreme_pain'])
    peristalsis = st.sidebar.selectbox('Peristalsis', ['absent', 'hypomotile', 'normal', 'hypermotile'])
    abdominal_distention = st.sidebar.selectbox('Abdominal Distention', ['slight', 'moderate', 'severe', 'none'])
    nasogastric_tube = st.sidebar.selectbox('Nasogastric Tube', ['slight', 'significant', 'none'])
    nasogastric_reflux = st.sidebar.selectbox('Nasogastric Reflux', ['none', 'more_1_liter', 'less_1_liter', 'absent'])
    nasogastric_reflux_ph = st.sidebar.number_input('Nasogastric Reflux pH', value=0.0)
    rectal_exam_feces = st.sidebar.selectbox('Rectal Exam Feces', ['normal', 'increased', 'decreased', 'absent'])
    abdomen = st.sidebar.selectbox('Abdomen', ['distend_large', 'distend_small', 'firm', 'none'])
    packed_cell_volume = st.sidebar.number_input('Packed Cell Volume', value=45)
    total_protein = st.sidebar.number_input('Total Protein', value=6.5)
    abdomo_appearance = st.sidebar.selectbox('Abdomo Appearance', ['cloudy', 'serosanguious', 'clear', 'none'])
    abdomo_protein = st.sidebar.number_input('Abdomo Protein', value=3.5)
    surgical_lesion = st.sidebar.selectbox('Surgical Lesion', ['yes', 'no'])
    lesion_1 = st.sidebar.number_input('Lesion 1', value=0)
    lesion_2 = st.sidebar.number_input('Lesion 2', value=0)
    lesion_3 = st.sidebar.number_input('Lesion 3', value=0)
    cp_data = st.sidebar.selectbox('CP Data', ['yes', 'no'])

# Create a DataFrame with the entered values
    user_input = pd.DataFrame({
        'id': 1,
        'surgery': [surgery],
        'age': [age],
        'rectal_temp': [rectal_temp],
        'pulse': [pulse],
        'respiratory_rate': [respiratory_rate],
        'temp_of_extremities': [temp_of_extremities],
        'peripheral_pulse': [peripheral_pulse],
        'mucous_membrane': [mucous_membrane],
        'capillary_refill_time': [capillary_refill_time],
        'pain': [pain],
        'peristalsis': [peristalsis],
        'abdominal_distention': [abdominal_distention],
        'nasogastric_tube': [nasogastric_tube],
        'nasogastric_reflux': [nasogastric_reflux],
        'nasogastric_reflux_ph': [nasogastric_reflux_ph],
        'rectal_exam_feces': [rectal_exam_feces],
        'abdomen': [abdomen],
        'packed_cell_volume': [packed_cell_volume],
        'total_protein': [total_protein],
        'abdomo_appearance': [abdomo_appearance],
        'abdomo_protein': [abdomo_protein],
        'surgical_lesion': [surgical_lesion],
        'lesion_1': [lesion_1],
        'lesion_2': [lesion_2],
        'lesion_3': [lesion_3],
        'cp_data': [cp_data]
    })

    print(preprocess_test_data(user_input).shape)
    # Preprocess the test data
    X_test_preprocessed, X_test_id = preprocess_test_data(user_input)
    # Display preprocessed data
    st.subheader('Preprocessed Test Data:')
    st.write(X_test_preprocessed)

    # Load the model
    model = load_model()

    # Make predictions
    predictions = model.predict(X_test_preprocessed)

    # Map predictions to labels
    predictions_labels = [prediction_mapping[prediction] for prediction in predictions]

    # Display predictions
    st.subheader('Predictions:')
    st.write(predictions_labels)
    # Display the shape of the predictions
    st.write(f"Predictions Shape: {len(predictions_labels)}")
