import streamlit as st 
import pandas as pd
from sklearn.linear_model import LogisticRegression 
import pickle

pickle_model=pickle.load(open("c:/Users/HP/Downloads/diabetes_model.pkl","rb"))


st.title('Model Deployment: Logistic Regression')
st.sidebar.header('User Input Parameters')

# ---------- USER INPUT ----------
def user_input_features():
    data = {
        "Pregnancies": st.sidebar.number_input("Number of Pregnancies", 0),
        "Glucose": st.sidebar.number_input("Glucose", 0),
        "BloodPressure": st.sidebar.number_input("Blood Pressure", 0),
        "SkinThickness": st.sidebar.number_input("Skin Thickness", 0),
        "Insulin": st.sidebar.number_input("Insulin", 0),
        "BMI": st.sidebar.number_input("BMI", 0.0),
        "DiabetesPedigreeFunction": st.sidebar.number_input("Diabetes Pedigree Function", 0.0),
        "Age": st.sidebar.number_input("Age", 0)
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()
st.subheader("User Input Parameters")
st.write(df)



# ---------- LOAD & TRAIN MODEL ----------
diabetes = pd.read_csv("diabetes.csv")
diabetes = diabetes.dropna()

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

clf = LogisticRegression()
clf.fit(X, y)


# ---------- PREDICTION ----------
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader("Predicted Result")
st.write("Diabetic" if prediction[0] == 1 else "Not Diabetic")

st.subheader("Prediction Probability")
st.write(prediction_proba)