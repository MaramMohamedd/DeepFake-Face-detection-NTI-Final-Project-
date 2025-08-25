import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

# --- Custom CSS Styling ---
def load_custom_css():
    """
    Load custom CSS for a futuristic AI-themed interface
    """
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: radial-gradient(ellipse at center, #0a0e27 0%, #020408 70%);
        font-family: 'Rajdhani', sans-serif;
        color: #ffffff;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #00d4ff 0%, #7b68ee 50%, #ff1493 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 50px rgba(0, 212, 255, 0.5);
        letter-spacing: 2px;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #8892b0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 400;
        margin-bottom: 3rem;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Futuristic card container */
    .ai-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(0, 212, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .ai-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        transition: left 0.8s ease;
    }
    
    .ai-card:hover::before {
        left: 100%;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 104, 238, 0.1) 100%);
        border: 2px dashed rgba(0, 212, 255, 0.5);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: rotate 8s linear infinite;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .upload-section:hover::before {
        opacity: 1;
    }
    
    .upload-section:hover {
        border-color: rgba(255, 20, 147, 0.8);
        background: linear-gradient(135deg, rgba(255, 20, 147, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.2);
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* MASSIVE prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-radius: 25px;
        padding: 4rem 3rem;
        margin: 3rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .prediction-real {
        border: 3px solid #00ff88;
        box-shadow: 
            0 0 50px rgba(0, 255, 136, 0.3),
            inset 0 0 50px rgba(0, 255, 136, 0.1);
    }
    
    .prediction-fake {
        border: 3px solid #ff0066;
        box-shadow: 
            0 0 50px rgba(255, 0, 102, 0.3),
            inset 0 0 50px rgba(255, 0, 102, 0.1);
    }
    
    .prediction-text {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .prediction-real .prediction-text {
        color: #00ff88;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.6);
    }
    
    .prediction-fake .prediction-text {
        color: #ff0066;
        text-shadow: 0 0 30px rgba(255, 0, 102, 0.6);
    }
    
    .confidence-display {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        opacity: 0.9;
    }
    
    /* Confidence bar */
    .confidence-bar-container {
        width: 100%;
        height: 20px;
        background: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
        margin: 2rem 0;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .confidence-bar-real {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        border-radius: 10px;
        transition: width 1.2s ease;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-bar-fake {
        height: 100%;
        background: linear-gradient(90deg, #ff0066, #ff1493);
        border-radius: 10px;
        transition: width 1.2s ease;
        box-shadow: 0 0 20px rgba(255, 0, 102, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-bar-real::after,
    .confidence-bar-fake::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Status messages */
    .success-msg {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 212, 255, 0.2));
        border: 1px solid #00ff88;
        color: #00ff88;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    
    .error-msg {
        background: linear-gradient(135deg, rgba(255, 0, 102, 0.2), rgba(255, 20, 147, 0.2));
        border: 1px solid #ff0066;
        color: #ff0066;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(255, 0, 102, 0.2);
    }
    
    .info-msg {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(123, 104, 238, 0.2));
        border: 1px solid #00d4ff;
        color: #00d4ff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(0, 212, 255, 0.2);
        border-top: 4px solid #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    .loading-text {
        color: #00d4ff;
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 2px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Image styling */
    .uploaded-image {
        border-radius: 20px;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.6),
            0 0 0 2px rgba(0, 212, 255, 0.2);
        transition: all 0.4s ease;
    }
    
    .uploaded-image:hover {
        transform: scale(1.02);
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.7),
            0 0 0 2px rgba(255, 20, 147, 0.4);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        color: #00d4ff;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-top: none;
        border-radius: 0 0 15px 15px;
        padding: 2rem;
    }
    
    /* Tech info styling */
    .tech-info {
        background: rgba(0, 0, 0, 0.3);
        border-left: 4px solid #00d4ff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-family: 'Orbitron', monospace;
        font-size: 0.9rem;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid rgba(0, 212, 255, 0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .prediction-text {
            font-size: 2rem;
        }
        
        .ai-card {
            padding: 1.5rem;
        }
        
        .upload-section {
            padding: 2rem;
        }
        
        .prediction-box {
            padding: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Functions ---

@st.cache_resource
def load_efficientnet_model(model_path):
    """
    Recreates the exact EfficientNetB4 model architecture used during training
    and then loads the weights from the file.
    """
    # Check if the model file exists before trying to load it
    if not os.path.exists(model_path):
        st.markdown('<div class="error-msg">🚨 SYSTEM ERROR: Model file not detected in current directory</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-msg">📡 Please ensure your model file is in the same directory as the script</div>', unsafe_allow_html=True)
        return None
    
    try:
        # Recreate the EXACT architecture you used during training
        base_model = EfficientNetB4(
            include_top=False, 
            weights=None,  # Don't load imagenet weights since we'll load your trained weights
            input_shape=(224, 224, 3)
        )
        
        # Build the model exactly as you did during training
        inputs = Input(shape=(224, 224, 3))
        
        # Apply EfficientNet preprocessing (like you did in training)
        x = preprocess_input(inputs)
        
        # Pass through base model (training=False to keep it in inference mode)
        x = base_model(x, training=False)
        
        # Add classification head (exactly as during training)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation="sigmoid")(x)  # You used sigmoid with 1 unit, not softmax with 2
        
        # Create the full model
        model = Model(inputs, outputs)
        
        # Load the trained weights from your saved file
        model.load_weights(model_path)

        st.markdown('<div class="success-msg">⚡ MODEL SUCCESSFULLY LOADED</div>', unsafe_allow_html=True)
        return model

    except Exception as e:
        st.markdown(f'<div class="error-msg">⚠️ SYSTEM MALFUNCTION: {e}</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-msg">🔧 The model file might be corrupted or incompatible with current TensorFlow version</div>', unsafe_allow_html=True)
        return None

def preprocess_image(image_file):
    """
    Preprocesses an image file uploaded in Streamlit for model prediction.
    """
    image = Image.open(image_file)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def make_prediction(model, preprocessed_image):
    """
    Makes a prediction using the loaded model.
    """
    predictions = model.predict(preprocessed_image)
    confidence = predictions[0][0]
    predicted_class = "FAKE" if confidence > 0.5 else "REAL"
    display_confidence = confidence if predicted_class == "FAKE" else (1 - confidence)
    
    return predicted_class, display_confidence, confidence

# --- Main Streamlit App Logic ---

def main():
    st.set_page_config(
        page_title="AI Deepfake Detection System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Load custom CSS
    load_custom_css()

    # Main title and subtitle
    st.markdown('<h1 class="main-title">🤖 AI DEEPFAKE DETECTION SYSTEM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced EfficientNet Neural Network • Real-time Face Authentication</p>', unsafe_allow_html=True)

    # Model loading
    model_paths = ["model_epoch_32.keras", "model_epoch_32.h5", "model_epoch_32.weights.h5"]
    model = None
    
    for path in model_paths:
        if os.path.exists(path):
            model = load_efficientnet_model(path)
            if model is not None:
                break
    
    if model is None:
        st.markdown('<div class="error-msg">🚨 CRITICAL ERROR: No compatible model file found</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-msg">📋 Supported formats: .keras, .h5, .weights.h5</div>', unsafe_allow_html=True)
        return

    # Upload section
    st.markdown('<h3 style="color: #00d4ff; font-family: Orbitron, monospace; text-align: center; margin: 3rem 0 2rem 0;">🎯 UPLOAD FACE FOR ANALYSIS</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select image file (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear face image for deepfake detection"
    )

    if uploaded_file is not None:
        try:
            # Create columns for layout
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                # Display uploaded image
                st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #00d4ff; font-family: Orbitron, monospace; margin-bottom: 1rem;">📷 INPUT IMAGE</h3>', unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True, caption="")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Analysis section
                st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #00d4ff; font-family: Orbitron, monospace; margin-bottom: 2rem;">🧠 NEURAL ANALYSIS</h3>', unsafe_allow_html=True)
                
                # Loading animation
                with st.spinner(""):
                    st.markdown('''
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">PROCESSING...</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Preprocess and predict
                    preprocessed_img = preprocess_image(uploaded_file)
                    predicted_class, display_confidence, raw_confidence = make_prediction(model, preprocessed_img)
                
                st.markdown('</div>', unsafe_allow_html=True)

            # MASSIVE PREDICTION RESULT BOX
            if predicted_class == "REAL":
                st.markdown(f'''
                <div class="prediction-box prediction-real">
                    <div class="prediction-text">✅ AUTHENTIC HUMAN FACE</div>
                    <div class="confidence-display">CONFIDENCE LEVEL: {display_confidence:.1%}</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-real" style="width: {display_confidence * 100}%"></div>
                    </div>
                    <div style="color: #00ff88; font-family: Rajdhani, sans-serif; font-size: 1.3rem; margin-top: 1rem;">
                        🛡️ VERIFIED REAL • NO MANIPULATION DETECTED
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
            else:
                st.markdown(f'''
                <div class="prediction-box prediction-fake">
                    <div class="prediction-text">⚠️ DEEPFAKE DETECTED</div>
                    <div class="confidence-display">CONFIDENCE LEVEL: {display_confidence:.1%}</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fake" style="width: {display_confidence * 100}%"></div>
                    </div>
                    <div style="color: #ff0066; font-family: Rajdhani, sans-serif; font-size: 1.3rem; margin-top: 1rem;">
                        🚨 AI MANIPULATION DETECTED • SYNTHETIC FACE
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Technical details expander
            with st.expander("🔬 DETAILED TECHNICAL ANALYSIS"):
                st.markdown(f'''
                <div class="tech-info">
                    <h4 style="color: #00d4ff; font-family: Orbitron, monospace;">MODEL'S OUTPUT</h4>
                    <p><strong>Raw Prediction Score:</strong> {raw_confidence:.6f}</p>
                    <p><strong>Classification Threshold:</strong> 0.500000</p>
                    <p><strong>Model Architecture:</strong> EfficientNetB4</p>
                    <p><strong>Input Resolution:</strong> 224×224 pixels</p>
                    <p><strong>Preprocessing:</strong> EfficientNet normalization</p>
                    <p><strong>Activation Function:</strong> Sigmoid (Binary Classification)</p>
                </div>
                
                <div class="tech-info">
                    <h4 style="color: #00d4ff; font-family: Orbitron, monospace;">PROBABILITY BREAKDOWN</h4>
                    <p><strong>Real Face Probability:</strong> {(1-raw_confidence)*100:.2f}%</p>
                    <p><strong>Deepfake Probability:</strong> {raw_confidence*100:.2f}%</p>
                    <p><strong>Decision Logic:</strong> {"FAKE (score > 0.5)" if raw_confidence > 0.5 else "REAL (score ≤ 0.5)"}</p>
                </div>
                ''', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-msg">💥 PROCESSING ERROR: {e}</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-msg">🔄 Try uploading a different image format or check image integrity</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('''
    <div class="footer">
        <p> POWERED BY EFFICIENTNET DEEP LEARNING ARCHITECTURE</p>
        <p> GRADUATE PROJECT • NTI</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()