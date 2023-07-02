import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load your dataset
df = pd.read_csv("thdata_final.csv")

# Let's divide the dataset into predictor features and target feature
X = df.drop('Group', axis=1)
y = df['Group']

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model_final = RandomForestClassifier(n_estimators=100,min_samples_split=2, min_samples_leaf= 4, max_features= 'sqrt' , max_depth= 10,random_state=42)
model_final.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model_final.predict(X_test)
st.write("""
# Simple Blood Test Predictor App
This app predicts the **Group**!
""")

st.sidebar.header('User Input Features')

def user_input_features():
    Age = st.sidebar.slider('Age', 0, 100, 30)
    Hb = st.sidebar.number_input('Hb', 0.0, 20.0, 10.0)
    MCH = st.sidebar.number_input('MCH', 0.0, 50.0, 25.0)
    MCHC = st.sidebar.number_input('MCHC', 0.0, 50.0, 25.0)
    RDW = st.sidebar.number_input('RDW', 0.0, 30.0, 15.0)
    RBCcount = st.sidebar.number_input('RBC count', 0.0, 10.0, 5.0)
    data = {'Age': Age,
            'Hb': Hb,
            'MCH': MCH,
            'MCHC': MCHC,
            'RDW': RDW,
            'RBC count': RBCcount}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Hard code the parameters of your trained model here
model_final = RandomForestClassifier(n_estimators=100,min_samples_split=2, min_samples_leaf= 4, max_features= 'sqrt' , max_depth= 10,random_state=42)
# Assume we have these fitted parameters:
model_final.fit(X, y)  # Your X, y values should be hard-coded here, not ideal!

prediction = model_final.predict(df)

st.subheader('Prediction')
group = np.array([1,2,3,4])
st.write(group[prediction])