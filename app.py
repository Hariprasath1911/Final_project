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
st.markdown(
    """
    <h1 style='color: green; font-family: "Arial", sans-serif; font-size: 40px; 
               text-shadow: 3px 3px 8px rgba(0,0,0,0.5); text-align: center;'>
        CNC : Real-time Data & Predictive Analytics
    </h1>
    """,
    unsafe_allow_html=True
)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
encoder=load_model("Encoder_MP.pkl")
scaler=load_model("scaler.pkl")
model = tf.keras.models.load_model("model_final.h5")
tab1, tab2 = st.tabs(["**Home**", "**Application**"])
with tab1:
    st.markdown("""
        **1. Introduction**
        In modern manufacturing, CNC (Computer Numerical Control) machines play a critical role in precision machining.
        However, unexpected failures and inefficiencies can lead to costly downtimes.
        My CNC Time Series Analysis project utilizes deep learning to analyze machine data, predict performance trends, and detect anomalies,
        ensuring optimal machine health and efficiency.
        
        **2. Problem Statement**
        CNC machines generate large volumes of time-series data, including spindle speed, feed rate, tool wear, and temperature variations.
        Manually monitoring these parameters is challenging and inefficient. Unscheduled downtime leads to production losses and increased maintenance costs.
        This project leverages deep learning models to identify patterns, detect anomalies, and predict potential failures, allowing for proactive maintenance.
        
        **3. Key Features**
        ✅ Real-Time Data Analysis – Processes CNC machine time-series data efficiently.
        ✅ Deep Learning Models – Implements LSTM/GRU-based models for precise forecasting.
        ✅ Anomaly Detection – Identifies unusual patterns to prevent machine failures.
        ✅ Feature Engineering – Extracts meaningful insights from historical machine logs.
        ✅ Interactive Dashboard – A Streamlit-based UI for visualizing trends and alerts.
        
        **4. Target Audience**
        🔹 Manufacturing Industries – Optimizing CNC machine performance and reducing downtime.
        🔹 Maintenance Teams – Predicting potential failures for proactive servicing.
        🔹 Industrial Data Analysts – Utilizing AI-driven insights for process optimization.
        
        **5. Technologies Used**
        🔸 Frontend: Streamlit for an intuitive and interactive web app.
        🔸 Backend: Python with libraries like TensorFlow/Keras, Pandas, and NumPy.
        🔸 Data Handling: Preprocessing using Pandas and Scikit-learn.
        
        **7. Conclusion**
        The CNC Time Series Analysis project is a data-driven approach to predictive maintenance in manufacturing.
        By leveraging deep learning, it enhances CNC machine efficiency, minimizes downtime, and provides actionable insights for maintenance teams.
        This project bridges the gap between AI and industrial automation, bringing intelligent decision-making to CNC operations.
        """)
with tab2:
    st.markdown(
    """
    <style>
    .stFileUploader label {
        color: green !important;  /* Change text color */
        font-size: 20px !important; /* Change font size */
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("**Upload an Excel file**", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, dtype=str)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
    
            if 'Machining_Process' in df.columns:
                df['Machining_Process'] = encoder.fit_transform(df[["Machining_Process"]])
            else:
                st.error("Column 'Machining_Process' not found in the uploaded file.")
            
        except Exception as e:
            st.error(f"Error: {e}")
        st.write(df)
    st.markdown(
    """
    <style>
    div.stButton > button {
        color: white !important;
        background-color: #008CBA !important; /* Blue color */
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #005f73 !important; /* Darker blue on hover */
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
    )        
    if st.button('Test Result'):
        df_scaled = scaler.fit_transform(df)
        prediction = model.predict(df_scaled)
        st.subheader("Predicted Test Result")
        df["Tool Wear"] = ["Worn" if p[0] > 0.5 else "Unworn" for p in prediction]
        df["Visual inspection"] = ["Properly Clamped" if p[1] > 0.5 else "Not Properly Clamped" for p in prediction]
        df["Machining Completion"] = ["Completed" if p[2] > 0.5 else "Not Completed" for p in prediction]
        st.write(df)
        st.markdown(
        """
        <style>
        div.stDownloadButton > button {
            color: white !important;  /* Text color */
            background-color: green !important;  /* Button background color */
            border-radius: 10px !important;  /* Rounded corners */
            font-size: 18px !important;  /* Font size */
            font-weight: bold !important; /* Bold text */
            padding: 10px 20px !important; /* Padding */
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

        tool_condition = ["Worn" if p[0] > 0.5 else "Unworn" for p in prediction]
        machine_finalized = ["Completed" if p[1] > 0.5 else "Not Completed" for p in prediction]
        visual_inspection = ["Inspection Passed" if p[2] > 0.5 else "Inspection Failed" for p in prediction]
        
        st.markdown(f"### :green[Tool Condition: {tool_condition[0]}]")
        st.markdown(f"### :green[Machine Finalized: {machine_finalized[0]}]")
        st.markdown(f"### :green[Visual Inspection: {visual_inspection[0]}]")

        #st.markdown(f"### :green[Tool Condition:{["Worn" if p[0][0] > 0.5 else "Unworn" for p in prediction]}]")
        #st.markdown(f"### :green[Machine Finalized:{["Completed" if p[1][0] > 0.5 else "Not Completed" for p in prediction]}]")
        #st.markdown(f"### :green[Visual Inspection:{["Inspection Passed" if p[2][0] > 0.5 else "Inspection Failed" for p in prediction]}]")
        
