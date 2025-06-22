import streamlit as st
import requests
from PIL import Image
import io
import json
from typing import Dict, Optional
import time

# Configuration
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/"

# Page configuration
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍓",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 5px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        height: 20px;
        margin: 5px 0;
    }
    .confidence-fill {
        background-color: #4CAF50;
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_image(image_bytes: bytes) -> Optional[Dict]:
    """Send image to API for prediction"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Please make sure the API server is running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def display_prediction(result: Dict):
    """Display prediction results in a nice format"""
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    
    # Main prediction
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎯 Prediction")
        st.markdown(f"## **{result['prediction']}**")
    with col2:
        st.markdown("### 📊 Confidence")
        st.markdown(f"## **{result['confidence']:.1%}**")
    
    # Probability distribution
    st.markdown("### 📈 All Probabilities")
    probabilities = result['probabilities']
    
    # Sort probabilities by value
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for fruit, prob in sorted_probs:
        col1, col2, col3 = st.columns([3, 5, 2])
        with col1:
            # Add emoji for each fruit
            emoji = {"Banana 1": "🍌", "Strawberry 1": "🍓", "Watermelon 1": "🍉"}.get(fruit, "🍎")
            st.write(f"{emoji} {fruit}")
        with col2:
            # Progress bar for probability
            st.progress(prob)
        with col3:
            st.write(f"{prob:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.title("🍓 Fruit Classifier")
    st.markdown("Upload an image to classify fruits (Banana, Strawberry, or Watermelon)")
    
    # Check API health
    api_status = check_api_health()
    
    if api_status:
        st.success("✅ API is running")
    else:
        st.error("❌ API is not responding. Please start the API server.")
        st.code("docker-compose up -d", language="bash")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a banana, strawberry, or watermelon"
    )
    
    # Two columns for image display and results
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("🔍 Classify Fruit", type="primary"):
            with st.spinner("Analyzing image..."):
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Get prediction
                result = predict_image(img_byte_arr)
                
                if result:
                    with col2:
                        display_prediction(result)
                    
                    # Add fun message based on confidence
                    if result['confidence'] > 0.9:
                        st.balloons()
                        st.success("🎉 Very confident prediction!")
                    elif result['confidence'] > 0.7:
                        st.info("👍 Good confidence level")
                    else:
                        st.warning("🤔 Low confidence - try a clearer image")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **Click 'Classify Fruit'** to get the prediction
        3. The model will identify if the image contains:
           - 🍌 Banana
           - 🍓 Strawberry
           - 🍉 Watermelon
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Ensure the fruit is the main subject
        - Avoid blurry or dark photos
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and AutoGluon | [API Docs]({}/docs)".format(API_URL))

if __name__ == "__main__":
    main()
