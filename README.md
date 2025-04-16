# Skin Condition Pre-screener

A machine learning application that analyzes images of skin conditions and provides preliminary classification and information. This tool can help with early detection of potential skin issues.

## Supported Conditions

- **Melanocytic Nevi (Moles)**: Common benign skin growths  
- **Melanoma**: A serious form of skin cancer  
- **Benign Keratosis**: Harmless skin growths that appear with age  
- **Basal Cell Carcinoma**: A type of skin cancer that grows slowly and is usually curable  
- **Actinic Keratoses**: Rough, scaly patches on the skin caused by years of sun exposure  
- **Vascular Lesions**: Abnormal blood vessels visible on the skin, such as hemangiomas  
- **Dermatofibroma**: Small, firm benign skin nodules often found on the lower legs  

## Quick Start

1. **Setup the environment**:
   `pip install -r requirements.txt`
2. **Download and prepare the dataset**:
   `python src/data_preparation.py`

3. **Train the model**:
  `python src/model_training.py`

4. **Run the Streamlit app**:
  `streamlit run app.py`

## Medical Disclaimer

This application is for educational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Dataset

This project uses the HAM10000 dataset, which contains dermatoscopic images of skin lesions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
