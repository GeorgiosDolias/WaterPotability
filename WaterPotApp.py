import streamlit as st
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.write("""
# Water Potability App
This app predicts if water is potable based on 9 variables.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    pH = st.sidebar.slider('pH', 0.22, 14.0, 7.0)
    Hardness = st.sidebar.slider('Hardness', 73, 318, 197)
    Solids = st.sidebar.slider('Solids', 320, 56500, 20930)
    Chloramines = st.sidebar.slider('Chloramines', 1.3, 13.12, 7.1 )
    Sulfate = st.sidebar.slider('Sulfate', 129, 481, 332)
    Conductivity = st.sidebar.slider('Conductivity', 201, 753, 423 )
    Organic_carbon = st.sidebar.slider('Organic carbon', 2.2, 27.0, 14.3)
    Trihalomethanes = st.sidebar.slider('Trihalomethanes', 8.57, 124.1, 66.54)
    Turbidity = st.sidebar.slider('Turbidity', 1.45, 6.5, 3.9)

    data = {        
        'pH': pH, 
        'Hardness':Hardness, 
        'Solids': Solids, 
        'Chloramines': Chloramines,
        'Sulfate': Sulfate, 
        'Conductivity': Conductivity, 
        'Organic_carbon': Organic_carbon, 
        'Trihalomethanes': Trihalomethanes, 
        'Turbidity': Turbidity
        }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

df_imported = pdf.read_csv("water_potability.csv")
df2 = df_imported.dropna()
water= df2.sample(frac=1)


X = water.iloc[:,:-1]
Y = water.iloc[:,-1]

sc = StandardScaler()
X = sc.fit_transform(X)

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

Outcome = numpy.array(['Not potable', "Potable"])

st.subheader('Class labels and their corresponding index number')
st.write(Outcome)

st.subheader('Prediction')
st.write(Outcome[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
