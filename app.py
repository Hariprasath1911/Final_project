import streamlit as st
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
import pickle
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU

def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: contain;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"12.png")

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
encoder=load_model("Encoder_MP.pkl")
scaler=load_model("scaler.pkl")
def load_model_file():
    custom_object1 = {"ReLU": ReLU}
    return load_model("model_final.h5", custom_objects=custom_object1)

model = load_model_file()

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        if 'Machining_Process' in df.columns:
            df['Machining_Process'] = encoder.transform(df["Machining_Process"])
        else:
            st.error("Column 'Machining_Process' not found in the uploaded file.")

        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        
    except Exception as e:
        st.error(f"Error: {e}")
if st.button('Predict'):
    prediction = model.predict(df_scaled)
    st.subheader("Predicted Car Price")
    st.markdown(f"### :green[{prediction}]")
