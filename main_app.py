import streamlit as st
import pickle
import numpy as np
import pandas as pd # For creating DataFrame for input
import matplotlib.pyplot as plt



# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered", # or "wide"
    initial_sidebar_state="expanded"
)


hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_container__1QSob {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Load Model and Scaler ---
# Option 1: Load the entire pipeline (Recommended)
PIPELINE_PATH = 'saved_model/diabetes_pipeline.pkl' # This contains both scaler and model
MODEL_CLASSIFIER_ONLY_PATH = 'saved_model/diabetes_model_classifier_only.pkl'
SCALER_PATH = 'saved_model/scaler.pkl'

# Try to load the pipeline first
try:
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    st.sidebar.success("Pipeline loaded successfully!")
    MODEL_LOAD_METHOD = "pipeline"
except FileNotFoundError:
    st.sidebar.warning(f"Pipeline file '{PIPELINE_PATH}' not found. Trying separate model and scaler...")
    try:
        with open(MODEL_CLASSIFIER_ONLY_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(SCALER_PATH, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        st.sidebar.success("Separate model and scaler loaded!")
        MODEL_LOAD_METHOD = "separate"
    except FileNotFoundError:
        st.sidebar.error(f"Critical Error: Neither pipeline nor separate model/scaler files found. Please ensure '{PIPELINE_PATH}' or ('{MODEL_CLASSIFIER_ONLY_PATH}' and '{SCALER_PATH}') exist.")
        st.stop() # Stop execution if models can't be loaded
    except Exception as e:
        st.sidebar.error(f"Error loading model/scaler: {e}")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading pipeline: {e}")
    st.stop()


# --- Application Title and Description ---
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of an individual having diabetes based on several health indicators. 
Please enter the patient's details in the sidebar.
""")

# --- Input Fields in Sidebar ---
st.sidebar.header("Patient Input Features")

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1, step=1,
                                          help="Number of times pregnant")
    glucose = st.sidebar.slider('Glucose (mg/dL)', min_value=0.0, max_value=250.0, value=120.0, step=0.1,
                                help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0.0, max_value=150.0, value=70.0, step=0.1,
                                       help="Diastolic blood pressure")
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', min_value=0.0, max_value=100.0, value=20.0, step=0.1,
                                       help="Triceps skin fold thickness")
    insulin = st.sidebar.slider('Insulin (mu U/ml)', min_value=0.0, max_value=900.0, value=80.0, step=0.1,
                                help="2-Hour serum insulin")
    bmi = st.sidebar.slider('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=32.0, step=0.1,
                            help="Body mass index (weight in kg / (height in m)^2)")
    dpf = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47, step=0.001,
                            help="A function that scores likelihood of diabetes based on family history")
    age = st.sidebar.number_input('Age (years)', min_value=1, max_value=120, value=30, step=1)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display Input Data ---
st.subheader('Patient Input Summary:')
st.write(input_df)

# --- Prediction ---
if st.sidebar.button('Predict Diabetes Status'):
    if MODEL_LOAD_METHOD == "pipeline":
        try:
            prediction_proba = pipeline.predict_proba(input_df) # Probabilities
            prediction = pipeline.predict(input_df) # Class (0 or 1)
        except Exception as e:
            st.error(f"Error during prediction with pipeline: {e}")
            st.stop()
    elif MODEL_LOAD_METHOD == "separate":
        try:
            # Ensure column order matches training if not using pipeline directly
            # The DataFrame creation above ensures this based on dictionary keys
            input_scaled = scaler.transform(input_df)
            prediction_proba = model.predict_proba(input_scaled)
            prediction = model.predict(input_scaled)
        except Exception as e:
            st.error(f"Error during prediction with separate model/scaler: {e}")
            st.stop()
    else: # Should not happen if checks above are correct
        st.error("Model not loaded correctly.")
        st.stop()


    st.subheader('Prediction Result:')
    
    if prediction[0] == 1:
        st.error('Prediction: **Diabetic** 😢')
    else:
        st.success('Prediction: **Non-Diabetic** 😊')

    st.subheader('Prediction Probability:')
    # Assuming binary classification, prediction_proba gives [[P(class 0), P(class 1)]]
    proba_df = pd.DataFrame({
        'Class': ['Non-Diabetic', 'Diabetic'],
        'Probability': prediction_proba[0]
    })
    st.write(proba_df)
    
    # Optional: Show a bar chart for probabilities
    fig, ax = plt.subplots()
    proba_df.set_index('Class').plot(kind='bar', ax=ax, legend=False, color=['skyblue', 'salmon'])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=0)
    st.pyplot(fig)

else:
    st.info("Click the 'Predict Diabetes Status' button in the sidebar to see the prediction.")

# --- Disclaimer ---
st.markdown("---")
st.info("**Disclaimer:** This app is for educational purposes only and should not be used as a substitute for professional medical advice. Consult with a healthcare provider for any health concerns.")


st.sidebar.markdown("---")

st.sidebar.markdown("Developed with 'Savi Gupta' using Streamlit.")
