# Data_Science_With_Python.
A collection of Python-based data science projects, exercises, and experiments as I learn and grow in the field.




import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Train the model directly
@st.cache_resource
def load_model():
    # Load and prepare data
    df = pd.read_csv('diabetes.csv')
    
    # Replace zeros with median
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_columns:
        df[col] = df[col].replace(0, df[col].median())
    
    # Features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, scaler

# Load model and scaler
with st.spinner('Loading model... please wait'):
    model, scaler = load_model()

st.title('🩺 Diabetes Risk Predictor')
st.write('Enter the patient details below to predict diabetes risk.')

# Input fields
Pregnancies = st.slider('Pregnancies', 0, 17, 1)
Glucose = st.slider('Glucose Level', 0, 200, 120)
BloodPressure = st.slider('Blood Pressure', 0, 122, 70)
SkinThickness = st.slider('Skin Thickness', 0, 99, 20)
Insulin = st.slider('Insulin Level', 0, 846, 80)
BMI = st.slider('BMI', 0.0, 67.1, 25.0)
DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
Age = st.slider('Age', 21, 81, 30)

if st.button('Predict'):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.error(f'⚠️ High Risk of Diabetes (Confidence: {prediction*100:.1f}%)')
    else:
        st.success(f'✅ Low Risk of Diabetes (Confidence: {(1-prediction)*100:.1f}%)')


req
streamlit
tf-keras
scikit-learn
pandas
numpy
matplotlib
seaborn