import streamlit as st
import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Add proper path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add src directory to the Python path
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Now import utils
from src.utils import predict_skin_condition, load_model, load_class_labels

# Page configuration
st.set_page_config(
    page_title="Skin Condition Pre-screener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling with dark mode
st.markdown("""
    <style>
    /* Base dark theme */
    body {
        color: #f0f0f0;
        background-color: #121212;
    }

    .stApp {
        background-color: #121212;
    }

    /* Streamlit container color */
    .css-18e3th9 {
        padding: 1rem;
        background-color: #121212;
    }

    /* Text colors */
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: #f0f0f0 !important;
    }

    /* Header styling */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #f0f0f0 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.1);
    }

    .subtitle {
        font-size: 1.5rem;
        font-weight: 400;
        margin-bottom: 2rem;
        color: #adb5bd !important;
    }

    /* Box styling */
    .disclaimer {
        background-color: rgba(128, 48, 48, 0.2);
        color: #f0c0c0 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #8b0000;
    }

    .disclaimer strong {
        color: #ff9999 !important;
    }

    .info-box {
        background-color: rgba(0, 64, 128, 0.2);
        color: #b3d9ff !important;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #004080;
    }

    .info-box strong, .info-box li strong {
        color: #b3d9ff !important;
    }

    .info-box ul {
        margin-left: 1.5rem;
        margin-top: 0.5rem;
    }

    .info-box li {
        color: #b3d9ff !important;
    }

    .result-box {
        background-color: rgba(40, 40, 45, 0.8);
        color: #f0f0f0 !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: 1px solid #404040;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .result-box h2, .result-box h4 {
        color: #f0f0f0 !important;
    }

    .result-box p {
        color: #d0d0d0 !important;
    }

    .high-confidence {
        color: #1e7b34 !important;
        background-color: rgba(40, 167, 69, 0.2);
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #28a745;
        display: inline-block;
    }

    .medium-confidence {
        color: #a07415 !important;
        background-color: rgba(255, 193, 7, 0.2);
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffc107;
        display: inline-block;
    }

    .low-confidence {
        color: #bd2130 !important;
        background-color: rgba(220, 53, 69, 0.2);
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #dc3545;
        display: inline-block;
    }

    /* Section headers */
    h3 {
        color: #f0f0f0 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #404040;
    }

    h4 {
        color: #d0d0d0 !important;
        margin-top: 1rem;
    }

    /* File uploader styling */
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px dashed #6c757d !important;
        padding: 1rem !important;
    }

    .stFileUploader > div > div > span {
        color: #adb5bd !important;
    }

    .stFileUploader > div > div > button {
        background-color: #375a7f !important;
        color: white !important;
        border: none !important;
    }

    /* Radio button styling */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 0.5rem;
    }

    .stRadio label span {
        color: #d0d0d0 !important;
    }

    /* Divider styling */
    hr {
        margin: 2rem 0;
        border-color: #404040;
    }

    /* Placeholder box */
    .placeholder-box {
        text-align: center;
        padding: 3rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        border: 1px dashed #6c757d;
    }

    .placeholder-text {
        color: #adb5bd !important;
        font-size: 1.2rem;
    }

    /* About section */
    .about-section {
        background-color: rgba(60, 65, 75, 0.3);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
        border: 1px solid #404040;
    }

    .about-section h3 {
        color: #f0f0f0 !important;
        border-bottom: 1px solid #505050;
        padding-bottom: 0.5rem;
    }

    .about-section p, .about-section li {
        color: #d0d0d0 !important;
    }

    /* Model path info */
    .model-path {
        font-size: 0.8rem;
        color: #808080 !important;
        background-color: rgba(60, 60, 60, 0.5);
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-family: monospace;
        margin-bottom: 1rem;
    }

    /* Image styling */
    img {
        border-radius: 0.5rem;
        border: 1px solid #404040;
    }

    /* Caption for images */
    .css-10trblm {
        color: #adb5bd !important;
        margin-top: 0.5rem;
    }

    /* Spinner */
    .stSpinner > div > div {
        border-color: #375a7f transparent transparent !important;
    }

    /* Error message */
    .element-container .stAlert {
        background-color: rgba(220, 53, 69, 0.2) !important;
        color: #ff9999 !important;
        border: 1px solid #dc3545 !important;
    }

    /* Warning message */
    .element-container .stWarning {
        background-color: rgba(255, 193, 7, 0.2) !important;
        color: #ffe066 !important;
        border: 1px solid #ffc107 !important;
    }

    /* Selection color */
    ::selection {
        background-color: #375a7f;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Load and cache the model and class labels."""
    # Define project base directory (root folder of project)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check for model files using the correct paths
    model_best_path = os.path.join(base_dir, 'models', 'skin_condition_model_best.h5')
    model_final_path = os.path.join(base_dir, 'models', 'skin_condition_model_final.h5')

    # Use a more subtle way to display the model path
    st.markdown(f'<div class="model-path">Alternative model location: {model_final_path}</div>', unsafe_allow_html=True)

    if not os.path.exists(model_best_path) and not os.path.exists(model_final_path):
        st.warning(f"No trained model found at either location. Please train the model first.")
        return None, None

    try:
        model = load_model()
        class_labels = load_class_labels()
        return model, class_labels
    except Exception as e:
        st.error(f"Error loading model or class labels: {e}")
        return None, None


def main():
    # Header
    st.markdown("<div class='title'>Skin Condition Pre-screener</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload an image of a skin condition for preliminary analysis</div>",
                unsafe_allow_html=True)

    # Medical disclaimer
    st.markdown("""
        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This application is for educational purposes only and is not intended to replace 
            professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified 
            health provider with any questions you may have regarding a medical condition.
        </div>
    """, unsafe_allow_html=True)

    # Load model and class labels
    model, class_labels = load_resources()

    # Main content area with more spacing
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<h3>Upload Image</h3>", unsafe_allow_html=True)

        # Information about supported conditions
        # Update the information box in the main() function to include 4 more conditions
        # Add this in the col1 section where the info-box is defined

        # Information about supported conditions (updated with 4 more conditions)
        st.markdown("""
            <div class="info-box">
                <strong>Supported Conditions:</strong>
                <ul>
                    <li><strong>Melanocytic Nevi (nv):</strong> Common moles</li>
                    <li><strong>Melanoma (mel):</strong> A type of skin cancer</li>
                    <li><strong>Benign Keratosis (bkl):</strong> Benign skin growths</li>
                    <li><strong>Basal Cell Carcinoma (bcc):</strong> Most common type of skin cancer</li>
                    <li><strong>Actinic Keratosis (akiec):</strong> Precancerous skin lesions</li>
                    <li><strong>Vascular Lesions (vasc):</strong> Blood vessel abnormalities</li>
                    <li><strong>Dermatofibroma (df):</strong> Benign skin nodules</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Image upload
        uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

        # Demo images option
        st.markdown("<h4>Or try a sample image:</h4>", unsafe_allow_html=True)
        demo_option = st.radio(
            "Sample Images",
            ("None", "Sample Melanocytic Nevi", "Sample Melanoma", "Sample Benign Keratosis"),
            index=0
        )

        # Use demo image if selected
        if demo_option != "None":
            # Get base directory
            base_dir = os.path.dirname(os.path.abspath(__file__))

            sample_images = {
                "Sample Melanocytic Nevi": os.path.join(base_dir, "samples", "nv_sample.jpg"),
                "Sample Melanoma": os.path.join(base_dir, "samples", "mel_sample.jpg"),
                "Sample Benign Keratosis": os.path.join(base_dir, "samples", "bkl_sample.jpg"),
                "Sample Basal Cell Carcinoma": os.path.join(base_dir, "samples", "bcc_sample.jpg"),
                "Sample Actinic Keratosis": os.path.join(base_dir, "samples", "akiec_sample.jpg"),
                "Sample Vascular Lesions": os.path.join(base_dir, "samples", "vasc_sample.jpg"),
                "Sample Dermatofibroma": os.path.join(base_dir, "samples", "df_sample.jpg")
            }

            sample_path = sample_images[demo_option]

            # Check if sample exists, if not, show error
            if not os.path.exists(sample_path):
                st.error(f"Sample image {sample_path} not found. Please upload your own image.")
            else:
                # Read the sample image
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()

                # Display the sample image
                st.image(image_bytes, caption=f"{demo_option}", width=300)

                # Create a placeholder for the uploaded file
                if uploaded_file is None:
                    uploaded_file = sample_path

    with col2:
        st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)

        if uploaded_file is not None:
            # Process the uploaded file
            if isinstance(uploaded_file, str):  # Sample image
                with open(uploaded_file, "rb") as f:
                    image_bytes = f.read()
            else:  # User uploaded image
                image_bytes = uploaded_file.read()
                st.image(image_bytes, caption="Uploaded Image", width=300)

            # Show analysis in progress
            if model is not None and class_labels is not None:
                with st.spinner("Analyzing image..."):
                    # Add a small delay to show the spinner
                    time.sleep(1)

                    # Make prediction
                    result = predict_skin_condition(image_bytes, model, class_labels)

                    # Determine confidence level class
                    confidence_class = ""
                    confidence_text = ""
                    confidence = result['confidence']

                    if confidence > 0.7:
                        confidence_class = "high-confidence"
                        confidence_text = "High Confidence"
                    elif confidence > 0.4:
                        confidence_class = "medium-confidence"
                        confidence_text = "Medium Confidence"
                    else:
                        confidence_class = "low-confidence"
                        confidence_text = "Low Confidence"

                    # Display results
                    st.markdown(f"""
                        <div class="result-box">
                            <h4>Detected Condition:</h4>
                            <h2>{result['condition_name']}</h2>
                            <div class="{confidence_class}">
                                {confidence_text} ({confidence:.1%})
                            </div>
                            <hr>
                            <h4>Description:</h4>
                            <p>{result['description']}</p>
                            <h4>Recommendation:</h4>
                            <p>{result['recommendation']}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Show probability distribution
                    st.markdown("<h4>Probability Distribution:</h4>", unsafe_allow_html=True)

                    # Create bar chart with dark mode styling
                    # Update the condition_names dictionary in the results section
                    condition_names = {
                        'nv': 'Melanocytic Nevi',
                        'mel': 'Melanoma',
                        'bkl': 'Benign Keratosis',
                        'bcc': 'Basal Cell Carcinoma',
                        'akiec': 'Actinic Keratosis',
                        'vasc': 'Vascular Lesions',
                        'df': 'Dermatofibroma'
                    }

                    labels = [condition_names.get(code, code) for code in result['all_probabilities'].keys()]
                    probs = list(result['all_probabilities'].values())

                    # Set up the figure with dark mode colors
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor('#121212')
                    ax.set_facecolor('#1e1e1e')

                    # Set different colors for each bar based on condition
                    colors = ['#2ecc71' if label == condition_names.get(result['condition_code'],
                                                                        result['condition_code']) else '#6c757d' for
                              label in labels]

                    # Create horizontal bar chart
                    bars = ax.barh(labels, probs, color=colors, height=0.6)

                    # Add percentage labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(max(width + 0.01, 0.05),
                                bar.get_y() + bar.get_height() / 2,
                                f'{width:.1%}',
                                va='center',
                                color='white',
                                fontsize=10)

                    # Set chart properties
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability', color='white', fontsize=12)
                    ax.set_title('Condition Probability Distribution', color='white', fontsize=14)
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')

                    # Add grid lines
                    ax.grid(axis='x', color='#333333', linestyle='--', alpha=0.3)

                    # Remove spines
                    for spine in ax.spines.values():
                        spine.set_color('#333333')

                    fig.tight_layout()

                    # Display the chart in Streamlit
                    st.pyplot(fig)
            else:
                st.error("Model not loaded. Please make sure the model is trained and available.")
        else:
            # Placeholder when no image is uploaded
            st.markdown("""
                <div class="placeholder-box">
                    <p class="placeholder-text">
                        Upload an image or select a sample to see analysis results
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # Additional information
    st.markdown("---")
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown("<h3>About This Tool</h3>", unsafe_allow_html=True)
    # Update the About section with all 7 conditions
    st.markdown("""
        This skin condition pre-screener uses a convolutional neural network trained on the HAM10000 dataset
        to identify seven common skin conditions:

        - **Melanocytic Nevi**: Common moles, generally benign but should be monitored
        - **Melanoma**: A serious form of skin cancer that requires medical attention
        - **Benign Keratosis**: Harmless skin growths that often appear with age
        - **Basal Cell Carcinoma**: The most common type of skin cancer, usually grows slowly
        - **Actinic Keratosis**: Precancerous skin lesions caused by sun damage
        - **Vascular Lesions**: Blood vessel abnormalities in the skin
        - **Dermatofibroma**: Benign nodules that often appear on the arms and legs

        **Important Note**: This tool is not a substitute for professional medical advice. If you have concerns
        about a skin condition, please consult a dermatologist.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()