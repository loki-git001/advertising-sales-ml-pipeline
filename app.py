import streamlit as st
import pandas as pd
import joblib
import os

# --- Layout and Configuration ---
st.set_page_config(
    page_title="Advertising Sales Predictor",
    page_icon="📈",
    layout="centered"
)

# --- Define Model Path ---
MODEL_PATH = "models/optimized_lasso_pipeline.joblib"

# --- Main App ---
st.title("📈 Advertising Sales Predictor")
st.markdown("""
Welcome to the Sales Forecasting Tool! 
Adjust the advertising budgets below to instantly predict upcoming sales.
Our backend leverages an optimized Lasso Regression algorithm.
""")

st.divider()

# --- Load Model safely ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

if pipeline is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Please run the training notebook and save the model first!")
else:
    # --- Input Interface ---
    st.subheader("Configure Campaign Budgets ($)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tv_spend = st.number_input("📺 TV Budget", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
        
    with col2:
        radio_spend = st.number_input("📻 Radio Budget", min_value=0.0, max_value=200.0, value=30.0, step=5.0)
        
    with col3:
        newspaper_spend = st.number_input("📰 Newspaper Budget", min_value=0.0, max_value=200.0, value=20.0, step=5.0)

    # --- Prediction Logic ---
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Calculate Expected Sales", type="primary", use_container_width=True)

    if predict_btn:
        # Create a dataframe to match the X format the pipeline was trained on
        input_data = pd.DataFrame([[tv_spend, radio_spend, newspaper_spend]], 
                                  columns=['TV', 'Radio', 'Newspaper'])
        
        # Predict using the loaded pipeline
        prediction = pipeline.predict(input_data)[0]
        
        st.success("Prediction complete!")
        
        # Display the result in a highly visible metric card
        st.metric(label="Predicted Sales Expected (Thousands)", value=f"{prediction:.2f}")
        
        # Adding a dynamic insightful note based on our EDA
        st.info("💡 **Insights Note**: Our model successfully zeroed out 'Newspaper' as statistical noise. Adjusting the Newspaper budget will not change the predicted sales outcome!")
