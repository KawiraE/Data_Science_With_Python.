import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

@st.cache_resource
def load_model():
    #df = pd.read_csv('diabetes.csv')
    base_path = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_path, 'diabetes.csv'))
    
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_columns:
        df[col] = df[col].replace(0, df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        max_iter=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, scaler

model, scaler = load_model()

st.title('🩺 Diabetes Risk Predictor')
st.write('Enter the patient details below to predict diabetes risk.')

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
    prediction = model.predict_proba(input_scaled)[0][1]

    if prediction > 0.5:
        st.error(f'⚠️ High Risk of Diabetes (Confidence: {prediction*100:.1f}%)')
    else:
        st.success(f'✅ Low Risk of Diabetes (Confidence: {(1-prediction)*100:.1f}%)')
