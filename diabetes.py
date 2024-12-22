import pickle
import streamlit as st
import numpy as np

# Load the trained model
with open('diabetes.sav', 'rb') as file:
    model = pickle.load(file)

# Define the front-end interface
st.title('Prediksi Diabetes')
st.write("""
Aplikasi ini memprediksi kemungkinan seseorang menderita diabetes berdasarkan masukan data medis. 
Silakan isi data di bawah ini untuk mendapatkan hasil prediksi.
""")

# Create a function to get user input
def get_user_input():
    features = {
        'Pregnancies': st.number_input('Pregnancies', min_value=0.0, value=1.0, step=1.0),
        'Glucose': st.number_input('Glucose', min_value=0.0, value=100.0, step=1.0),
        'Blood Pressure': st.number_input('Blood Pressure', min_value=0.0, value=80.0, step=1.0),
        'Skin Thickness': st.number_input('Skin Thickness', min_value=0.0, value=20.0, step=1.0),
        'Insulin': st.number_input('Insulin', min_value=0.0, value=80.0, step=1.0),
        'BMI': st.number_input('BMI', min_value=0.0, value=25.0, step=0.1),
        'Diabetes Pedigree Function': st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5, step=0.01),
        'Age': st.number_input('Age', min_value=0, value=30, step=1)
    }
    return np.array(list(features.values()))

# Get user input
input_data = get_user_input()

# Make predictions
if st.button('Prediksi'):
    try:
        # Reshape input for prediction
        input_data = input_data.reshape(1, -1)
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]  # Probabilitas kelas positif (1)
        
        # Display results
        if prediction[0] == 1:
            st.error(f"Hasil: Anda kemungkinan menderita diabetes dengan probabilitas {probability:.2f}.")
        else:
            st.success(f"Hasil: Anda kemungkinan besar tidak menderita diabetes dengan probabilitas {probability:.2f}.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
