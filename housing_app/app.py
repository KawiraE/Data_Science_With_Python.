import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Train the model directly
@st.cache_resource
def load_model():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), X.columns.tolist())
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(n_neighbors=9, weights='distance', p=1))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

model = load_model()

st.title('🏠 California House Price Predictor')
st.write('Enter the housing features below to predict the median house value.')

MedInc = st.slider('Median Income (in $10,000s)', 0.5, 15.0, 3.0)
HouseAge = st.slider('House Age (years)', 1.0, 52.0, 20.0)
AveRooms = st.slider('Average Rooms', 1.0, 10.0, 5.0)
AveBedrms = st.slider('Average Bedrooms', 1.0, 5.0, 1.0)
Population = st.slider('Population', 3.0, 3500.0, 1000.0)
AveOccup = st.slider('Average Occupancy', 1.0, 6.0, 3.0)
Latitude = st.slider('Latitude', 32.0, 42.0, 36.0)
Longitude = st.slider('Longitude', -124.0, -114.0, -119.0)

if st.button('Predict Price'):
    import pandas as pd
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]],
                            columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                                     'Population', 'AveOccup', 'Latitude', 'Longitude'])
    prediction = model.predict(input_data)
    st.success(f'Predicted House Value: ${prediction[0]*100000:,.0f}')