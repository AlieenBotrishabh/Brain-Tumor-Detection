from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Global variables
model = None
target_size = None

def load_h5_model():
    """Load the .h5 model file and extract target size"""
    global model, target_size
    try:
        # Load model - update path as needed
        model = load_model('brain_tumor_model.h5')
        
        # Alternative: Download from Hugging Face
        # from huggingface_hub import hf_hub_download
        # model_path = hf_hub_download(repo_id="Ace6868/brain-tumor-detection", filename="model.h5")
        # model = load_model(model_path)
        
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Extract target size from model input shape
        input_shape = model.input_shape
        if len(input_shape) == 4:  # (batch, height, width, channels)
            height, width = input_shape[1], input_shape[2]
            target_size = (width, height)  # PIL/cv2 expects (width, height)
            print(f"Detected target size: {target_size}")
        else:
            target_size = (128, 128)  # fallback
            print(f"Using fallback target size: {target_size}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(image):
    """
    Preprocess image for the model using detected target size
    """
    try:
        print(f"Original image mode: {image.mode}, size: {image.size}")
        
        # Convert to grayscale if not already
        if image.mode != 'L':  # 'L' mode is grayscale
            image_gray = image.convert('L')
        else:
            image_gray = image
        
        print(f"After grayscale conversion: {image_gray.mode}, size: {image_gray.size}")
        
        # Resize image using the detected target size
        image_resized = image_gray.resize(target_size, Image.LANCZOS)
        print(f"Resized to: {image_resized.size}")
        
        # Convert to numpy array
        img_array = np.array(image_resized, dtype=np.float32)
        print(f"Numpy array shape: {img_array.shape}")
        
        # Normalize to 0-1 range
        img_array = img_array / 255.0
        
        # Add channel dimension: (height, width) -> (height, width, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        print(f"After adding channel dimension: {img_array.shape}")
        
        # Add batch dimension: (height, width, 1) -> (1, height, width, 1)
        img_batch = np.expand_dims(img_array, axis=0)
        print(f"Final shape for model: {img_batch.shape}")
        
        return img_batch
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")

def make_prediction(processed_image):
    """Make prediction using the model"""
    try:
        print(f"Input shape to model: {processed_image.shape}")
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions: {predictions}")
        
        # Process predictions based on output format
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            # Binary classification with single output
            confidence = float(predictions[0][0])
            # Sigmoid output: > 0.5 means positive class (tumor)
            is_tumor = confidence > 0.5
            label = "tumor" if is_tumor else "no_tumor"
        elif len(predictions.shape) == 2 and predictions.shape[1] == 2:
            # Binary classification with 2 outputs (softmax)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            class_labels = ["no_tumor", "tumor"]
            label = class_labels[predicted_class]
            is_tumor = predicted_class == 1
        else:
            # Handle other cases
            confidence = float(np.max(predictions))
            is_tumor = confidence > 0.5
            label = "tumor" if is_tumor else "no_tumor"
        
        return {
            'is_tumor': is_tumor,
            'label': label,
            'confidence': confidence,
            'raw_predictions': predictions.tolist()
        }
        
    except Exception as e:
        raise Exception(f"Error making prediction: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=== NEW PREDICTION REQUEST ===")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        print(f"Processing file: {file.filename}")
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        result = make_prediction(processed_image)
        
        # Format response
        response = {
            'prediction': 'Tumor Detected' if result['is_tumor'] else 'No Tumor',
            'label': result['label'],
            'confidence': f"{result['confidence']:.2%}",
            'confidence_score': result['confidence']
        }
        
        print(f"Final response: {response}")
        print("=== PREDICTION COMPLETE ===")
        return jsonify(response)
        
    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        result = make_prediction(processed_image)
        
        # Format response
        response = {
            'prediction': 'Tumor Detected' if result['is_tumor'] else 'No Tumor',
            'label': result['label'],
            'confidence': f"{result['confidence']:.2%}",
            'confidence_score': result['confidence']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_model', methods=['GET'])
def test_model():
    """Test endpoint to verify model loading and input shape"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Create a dummy input with the correct detected size
        height, width = target_size[1], target_size[0]  # target_size is (width, height)
        dummy_input = np.random.random((1, height, width, 1)).astype(np.float32)
        
        print(f"Testing with input shape: {dummy_input.shape}")
        dummy_prediction = model.predict(dummy_input, verbose=0)
        print(f"Test prediction successful! Output shape: {dummy_prediction.shape}")
        
        return jsonify({
            'status': 'Model working perfectly!',
            'model_input_shape': str(model.input_shape),
            'model_output_shape': str(model.output_shape),
            'detected_target_size': target_size,
            'test_input_shape': str(dummy_input.shape),
            'test_output_shape': str(dummy_prediction.shape),
            'sample_prediction': dummy_prediction.tolist()
        })
        
    except Exception as e:
        print(f"Test model error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    model_info = {}
    
    if model is not None:
        model_info = {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'detected_target_size': target_size
        }
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_info': model_info
    })

if __name__ == '__main__':
    print("Loading model...")
    if load_h5_model():
        print(f"Model loaded successfully with target size: {target_size}")
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check the model path.")