import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("🍷 Wine Quality Prediction App")
st.write("Εισάγετε τα φυσικοχημικά χαρακτηριστικά του κρασιού για πρόβλεψη ποιότητας (0–10).")

# Φόρτωση scaler/pca/model
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = load_model('wine_nn_model.h5')

# Είσοδοι από το χρήστη
st.sidebar.header("Χαρακτηριστικά κρασιού")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.08)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
density = st.sidebar.number_input("Density", 0.990, 1.005, 0.997)
pH = st.sidebar.number_input("pH", 2.5, 4.5, 3.3)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.6)
alcohol = st.sidebar.number_input("Alcohol", 5.0, 15.0, 10.0)

# Δημιουργία της εισόδου:
x = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
               chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
               density, pH, sulphates, alcohol]])

# Προεπεξεργασία / χρήση των pca&scaler
x_scaled = scaler.transform(x)
x_pca = pca.transform(x_scaled)

# Τελική Πρόβλεψη
if st.button("🔮 Πρόβλεψη Ποιότητας"):
    pred_probs = model.predict(x_pca)
    predicted_class = np.argmax(pred_probs, axis=1)[0]
    confidence = np.max(pred_probs)
    st.success(f"**Προβλεπόμενη ποιότητα:** {predicted_class}/10  \n(Εμπιστοσύνη: {confidence:.2%})")

