from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
import numpy as np
import librosa
import joblib
import os
import sys
import io
import base64
import serial
import json
from flask import jsonify

# Set matplotlib backend before importing pyplot to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from werkzeug.utils import secure_filename
import soundfile as sf

# Add src to path
import sys
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'src'))

try:
    from src.hybrid_model import HybridCryClassifier
    from src.data_preprocessing import extract_all_features
except ImportError:
    try:
        from hybrid_model import HybridCryClassifier
        from data_preprocessing import extract_all_features
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Python path: {sys.path[:3]}")
        raise

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
label_encoder = None
device = None

def load_model():
    """Load the trained model and necessary components"""
    global model, label_encoder, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load label encoder using joblib (consistent with training)
        label_encoder = joblib.load('../models/label_encoder.pkl')
        
        # Load processed data to get feature dimensions
        norm_data = np.load('../models/processed_data.npz')
        
        # Get feature dimensions from the saved data
        mfcc_dim = norm_data['norm_mfcc_mean'].shape[0]
        spectral_dim = norm_data['norm_spectral_mean'].shape[0] 
        rhythm_dim = norm_data['norm_rhythm_mean'].shape[0]
        
        # Load model
        model_path = '../models/hybrid_model.pth'
        if os.path.exists(model_path):
            # Get spectrogram shape from checkpoint or use default
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model instance with correct parameters
            model = HybridCryClassifier(
                num_classes=len(label_encoder.classes_),
                mfcc_dim=mfcc_dim,
                spectral_dim=spectral_dim,
                rhythm_dim=rhythm_dim,
                spectrogram_shape=(128, 313),  # Default from training
                hidden_dim=128,
                lstm_layers=1
            )
            
            # Load trained weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            
            print("Model loaded successfully!")
            print(f"Model classes: {label_encoder.classes_}")
            print(f"Feature dimensions - MFCC: {mfcc_dim}, Spectral: {spectral_dim}, Rhythm: {rhythm_dim}")
            return True
        else:
            print("Model file not found!")
            return False
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_audio_file(file_path):
    """Preprocess audio file for prediction"""
    try:
        # Extract all features using standalone function
        mfcc_feat, spectral_feat, rhythm_feat, spectrogram, raw_audio = extract_all_features(file_path)
        
        # Load normalization parameters
        norm_data = np.load('../models/processed_data.npz')
        
        # Normalize features
        mfcc_feat = (mfcc_feat - norm_data['norm_mfcc_mean']) / norm_data['norm_mfcc_std']
        spectral_feat = (spectral_feat - norm_data['norm_spectral_mean']) / norm_data['norm_spectral_std']
        rhythm_feat = (rhythm_feat - norm_data['norm_rhythm_mean']) / norm_data['norm_rhythm_std']
        spectrogram = (spectrogram - norm_data['norm_spec_mean']) / norm_data['norm_spec_std']
        
        return {
            'raw_audio': raw_audio,
            'mfcc': mfcc_feat,
            'spectral': spectral_feat,
            'rhythm': rhythm_feat,
            'spectrogram': spectrogram
        }
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def predict_cry_type(features):
    """Predict cry type from features"""
    try:
        with torch.no_grad():
            # Convert to tensors (matching the model's expected input format)
            raw_audio = torch.FloatTensor(features['raw_audio']).unsqueeze(0).to(device)
            spectrogram = torch.FloatTensor(features['spectrogram']).unsqueeze(0).to(device)  # Already 2D
            mfcc = torch.FloatTensor(features['mfcc']).unsqueeze(0).to(device)
            spectral = torch.FloatTensor(features['spectral']).unsqueeze(0).to(device)
            rhythm = torch.FloatTensor(features['rhythm']).unsqueeze(0).to(device)
            
            # Make prediction (model returns tuple: outputs, None)
            outputs, _ = model(raw_audio, spectrogram, mfcc, spectral, rhythm)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get class name
            class_name = label_encoder.classes_[predicted_class]
            
            # Get all class probabilities
            class_probs = {}
            for i, class_label in enumerate(label_encoder.classes_):
                class_probs[class_label] = probabilities[0][i].item()
            
            return {
                'predicted_class': class_name,
                'confidence': confidence,
                'probabilities': class_probs
            }
    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_prediction_chart(probabilities):
    """Create a bar chart of prediction probabilities"""
    plt.figure(figsize=(10, 6))
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = plt.bar(classes, probs, color=colors[:len(classes)])
    
    plt.title('Cry Classification Probabilities', fontsize=16, fontweight='bold')
    plt.xlabel('Cry Type', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_string = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_string

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess audio
            features = preprocess_audio_file(file_path)
            if features is None:
                return jsonify({'error': 'Error processing audio file'})
            
            # Make prediction
            result = predict_cry_type(features)
            if result is None:
                return jsonify({'error': 'Error making prediction'})
            
            # Create visualization
            chart_img = create_prediction_chart(result['probabilities'])
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'chart': chart_img
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file format. Please upload WAV, MP3, FLAC, or M4A files.'})

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')

@app.route('/api/model_info')
def model_info():
    """API endpoint for model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    info = {
        'classes': label_encoder.classes_.tolist() if label_encoder else [],
        'model_type': 'Hybrid BiLSTM + Wav2Vec + Attention',
        'features': ['MFCC', 'Spectral', 'Rhythm', 'Raw Audio', 'Spectrogram'],
        'device': str(device) if device else 'Unknown'
    }
    
    return jsonify(info)

@app.route('/live-health-data')
def live_health_data():
    try:
        ser = serial.Serial('COM3', 115200, timeout=2)
        line = ser.readline().decode().strip()
        ser.close()
        data = json.loads(line)
        # Replace nulls with '--'
        for key in ['bpm', 'spo2', 'ta', 'to']:
            if data.get(key) is None:
                data[key] = '--'
        return jsonify(data)
    except Exception as e:
        return jsonify({
            'bpm': '--',
            'spo2': '--',
            'ta': '--',
            'to': '--',
            'finger': False,
            'error': str(e)
        })

# Add this to your Flask app (app.py or main.py)
@app.route('/model-stats')
def model_stats():
    # Replace these with your actual model metrics
    return {
        'accuracy': 0.85,  # Replace with your actual accuracy (0.0-1.0)
        'avg_inference_time': 180  # Replace with actual inference time in ms
    }

@app.route('/live-data')
def live_data():
    return render_template('live_data.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Baby Cry Classification Web App...")
    
    # Load model
    if load_model():
        print("Model loaded successfully!")
        app.run(debug=False)
    else:
        print("Failed to load model. Please ensure the model is trained and saved.")
        print("Run train_model.py first to train the model.")