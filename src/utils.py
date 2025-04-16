import cv2
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image
import io
import gc

# Get the base directory of your project
# If utils.py is in the src folder, this points to the parent of src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(BASE_DIR)

def load_class_labels():
    """Load the class labels mapping from the saved JSON file."""
    labels_path = os.path.join(BASE_DIR, 'models', 'class_labels.json')
    try:
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
        return class_labels
    except FileNotFoundError:
        print(f"Class labels file not found at: {labels_path}")
        raise


def load_model():
    """Load the trained model with memory optimizations."""
    # Set memory growth for any available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # Optimize CPU thread usage
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    # Load trained model with proper path resolution
    model_path = os.path.join(BASE_DIR, 'models', 'skin_condition_model_best.h5')
    fallback_path = os.path.join(BASE_DIR, 'models', 'skin_condition_model_final.h5')

    print(f"Looking for model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Best model not found, trying fallback: {fallback_path}")
        model_path = fallback_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path} or {fallback_path}. Please train the model first.")

    # Load model with reduced precision for inference if possible
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        # Compile with minimal options for inference only
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    return model


def get_model_input_size():
    """Get the input size expected by the model."""
    # Check if a model info file exists
    info_path = os.path.join(BASE_DIR, 'models', 'model_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model_info = json.load(f)
            return tuple(model_info.get('input_size', (64, 64)))

    # Default size for our optimized model
    return (64, 64)


def preprocess_image(image_bytes, target_size=None):
    """
    Preprocess an image for the model.

    Args:
        image_bytes: The image file as bytes
        target_size: The target image dimensions (will use model's default if not specified)

    Returns:
        The preprocessed image as a numpy array
    """
    # Get target size if not specified
    if target_size is None:
        target_size = get_model_input_size()

    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (in case it's not)
        image = image.convert('RGB')

        # Resize image
        image = image.resize(target_size)

        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0

        # Expand dimensions to match the model's expected input
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise


def get_condition_description(condition_code):
    """
    Get a description of the skin condition based on the code.

    Args:
        condition_code: The code of the condition (nv, mel, bkl, etc.)

    Returns:
        A description of the condition
    """
    descriptions = {
        'nv': {
            'name': 'Melanocytic Nevi (Mole)',
            'description': 'A benign growth of melanocytes (pigment cells). Most moles are harmless, but they should be monitored for changes.',
            'recommendation': 'Monitor for changes in size, shape, color, or if itching/bleeding occurs. Regular self-examination is recommended.'
        },
        'mel': {
            'name': 'Melanoma',
            'description': 'A serious form of skin cancer that can be life-threatening. Early detection is crucial for successful treatment.',
            'recommendation': 'Please consult a dermatologist immediately for proper diagnosis and treatment.'
        },
        'bkl': {
            'name': 'Benign Keratosis',
            'description': 'A common benign skin growth that appears as waxy, scaly, slightly raised bumps. They\'re harmless and typically appear with age.',
            'recommendation': 'Generally no treatment is necessary. Consult a dermatologist if the lesion becomes irritated or for cosmetic concerns.'
        },
        'bcc': {
            'name': 'Basal Cell Carcinoma',
            'description': 'The most common type of skin cancer. It rarely spreads to other parts of the body but can cause damage to surrounding tissue.',
            'recommendation': 'Please consult a dermatologist for proper diagnosis and treatment options.'
        },
        'akiec': {
            'name': 'Actinic Keratosis / Intraepithelial Carcinoma',
            'description': 'A pre-cancerous growth that can develop into squamous cell carcinoma if left untreated.',
            'recommendation': 'Medical evaluation is recommended. Treatment options include cryotherapy, topical medications, or other procedures.'
        },
        'vasc': {
            'name': 'Vascular Lesion',
            'description': 'Abnormalities of blood vessels, including hemangiomas, port-wine stains, and other vascular malformations.',
            'recommendation': 'Most vascular lesions are benign, but a dermatologist can provide proper evaluation and treatment options if needed.'
        },
        'df': {
            'name': 'Dermatofibroma',
            'description': 'A common benign skin growth that usually appears as a small, firm bump, often on the legs.',
            'recommendation': 'Generally no treatment is necessary. If the lesion changes or causes discomfort, consult a dermatologist.'
        }
    }

    return descriptions.get(condition_code, {
        'name': 'Unknown Condition',
        'description': 'This condition could not be identified with confidence.',
        'recommendation': 'Please consult a dermatologist for proper diagnosis.'
    })


def predict_skin_condition(image_bytes, model=None, class_labels=None):
    """
    Predict the skin condition from an image with memory optimization.

    Args:
        image_bytes: The image file as bytes
        model: The trained model (will be loaded if not provided)
        class_labels: The class labels (will be loaded if not provided)

    Returns:
        A dictionary with prediction results
    """
    try:
        # Load model and class labels if not provided
        if model is None:
            model = load_model()

        if class_labels is None:
            class_labels = load_class_labels()

        # Preprocess the image
        processed_image = preprocess_image(image_bytes)

        # Make prediction with reduced batch size for memory efficiency
        with tf.device('/cpu:0'):  # Force CPU for prediction
            predictions = model.predict(processed_image, batch_size=1, verbose=0)[0]

        # Get the top prediction
        top_prediction_idx = np.argmax(predictions)
        top_prediction_confidence = float(predictions[top_prediction_idx])

        # Get the condition code
        condition_code = class_labels[str(top_prediction_idx)]

        # Get condition information
        condition_info = get_condition_description(condition_code)

        # Create result dictionary
        result = {
            'condition_code': condition_code,
            'condition_name': condition_info['name'],
            'confidence': top_prediction_confidence,
            'description': condition_info['description'],
            'recommendation': condition_info['recommendation'],
            'all_probabilities': {class_labels[str(i)]: float(prob) for i, prob in enumerate(predictions)}
        }

        # Clean up to free memory
        gc.collect()

        return result

    except Exception as e:
        error_result = {
            'error': str(e),
            'condition_name': 'Error during prediction',
            'recommendation': 'Please try again with a different image or contact support.'
        }
        return error_result


def batch_predict(image_paths, batch_size=4):
    """
    Predict skin conditions for multiple images with memory-efficient batching.

    Args:
        image_paths: List of paths to image files
        batch_size: Size of batches to process at once

    Returns:
        List of prediction results
    """
    # Load model and labels once
    model = load_model()
    class_labels = load_class_labels()

    results = []

    # Process in small batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_results = []

        for path in batch_paths:
            try:
                with open(path, 'rb') as f:
                    image_bytes = f.read()
                result = predict_skin_condition(image_bytes, model, class_labels)
                result['image_path'] = path
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                batch_results.append({
                    'image_path': path,
                    'error': str(e)
                })

        results.extend(batch_results)

        # Force memory cleanup after each batch
        gc.collect()

    return results


def save_model_info(input_size=(64, 64)):
    """Save model metadata for later use during inference."""
    model_info = {
        'input_size': input_size,
        'model_version': '1.0',
        'optimization': 'cpu_optimized'
    }

    info_path = os.path.join(BASE_DIR, 'models', 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f)

    print("Model info saved.")