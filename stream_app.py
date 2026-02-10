import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("abalone_model.pkl")

## Streamlit app
st.title("Abalone Age Prediction")
st.write("Predict the age of an abalone based on physical measurements")

## Add information section
with st.expander("About this app"):
    st.write("""
    This app predicts the **age** of an abalone (measured by number of rings).
    
    **Features used:**
    - Physical dimensions: Length, Diameter, Height
    - Weight measurements: Whole weight, Shucked weight, Viscera weight, Shell weight
    
    **Model:** Tuned Gradient Boosting Regressor
    """)

## Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Dimensions")
    
    ## User inputs for dimensions
    length = st.slider("Length (mm)", 
                      min_value=0.0, 
                      max_value=1.0, 
                      value=0.5,
                      step=0.01,
                      help="Longest shell measurement")
    
    diameter = st.slider("Diameter (mm)", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.4,
                        step=0.01,
                        help="Perpendicular to length")
    
    height = st.slider("Height (mm)", 
                      min_value=0.0, 
                      max_value=1.0, 
                      value=0.15,
                      step=0.01,
                      help="With meat in shell")

with col2:
    st.subheader("Weight Measurements")
    
    ## User inputs for weights
    whole_weight = st.slider("Whole Weight (grams)", 
                            min_value=0.0, 
                            max_value=3.0, 
                            value=0.8,
                            step=0.01,
                            help="Whole abalone weight")
    
    shucked_weight = st.slider("Shucked Weight (grams)", 
                              min_value=0.0, 
                              max_value=2.0, 
                              value=0.35,
                              step=0.01,
                              help="Weight of meat")
    
    viscera_weight = st.slider("Viscera Weight (grams)", 
                              min_value=0.0, 
                              max_value=1.0, 
                              value=0.18,
                              step=0.01,
                              help="Gut weight (after bleeding)")
    
    shell_weight = st.slider("Shell Weight (grams)", 
                            min_value=0.0, 
                            max_value=1.5, 
                            value=0.24,
                            step=0.01,
                            help="Weight after drying")

## Predict button
if st.button("Predict Age", type="primary"):
    
    ## Calculate ratio features (same as training)
    shell_ratio = shell_weight / (whole_weight + 1e-8)
    shucked_ratio = shucked_weight / (whole_weight + 1e-8)
    
    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'Length': [length],
        'Diameter': [diameter],
        'Height': [height],
        'Whole weight': [whole_weight],
        'Shucked weight': [shucked_weight],
        'Viscera weight': [viscera_weight],
        'Shell weight': [shell_weight],
        'Shell_Ratio': [shell_ratio],
        'Shucked_Ratio': [shucked_ratio]
    })
    
    ## Predict
    y_pred = model.predict(df_input)[0]
    
    ## Display prediction
    st.success(f"### Predicted Number of Rings: **{y_pred:.1f}**")
    st.info(f"**Estimated Age: {y_pred + 1.5:.1f} years**  \n(Age = Rings + 1.5)")
    
    ## Show input summary
    with st.expander("Input Summary"):
        st.dataframe(df_input.T, use_container_width=True)

## Add footer with tips
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>💡 Tips:</b> Larger abalones typically have more rings. Adjust the measurements to see how age predictions change!</p>
</div>
""", unsafe_allow_html=True)

## Page design - Ocean theme background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }
    
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stSuccess, .stInfo {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)