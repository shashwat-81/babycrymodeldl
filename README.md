# 🍼 Baby Cry Classification with Deep Learning

A hybrid deep learning model for classifying baby cries into 5 categories: belly pain, cold/hot, discomfort, hungry, and tired.

## 🚀 Features

- **Hybrid AI Model**: Statistical features + CNN + Traditional ML
- **Multi-Feature Analysis**: MFCC, spectrograms, spectral, rhythm, and raw audio stats
- **Data Balancing**: SMOTE technique to handle class imbalance  
- **Data Augmentation**: Conservative audio augmentation for robust training
- **Web Interface**: Flask-based web application for easy use
- **Real-time Prediction**: Upload audio files and get instant results
- **Visualization**: Interactive charts showing prediction probabilities

## 📁 Project Structure

```
babycrymodeldl/
├── Data/                       # Dataset folder
│   ├── belly pain/            # Belly pain cry samples
│   ├── cold_hot/              # Cold/hot discomfort samples
│   ├── discomfort/            # General discomfort samples
│   ├── hungry/                # Hungry cry samples
│   └── tired/                 # Tired cry samples
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data preprocessing and feature extraction
│   ├── hybrid_model.py        # Hybrid deep learning model
│   └── data_loader.py         # Data loading and augmentation
├── models/                    # Trained models and artifacts
├── web_app/                   # Flask web application
│   ├── app.py                 # Flask application
│   ├── templates/             # HTML templates
│   └── static/                # Static files
├── test_dataset/              # Test samples for evaluation
├── requirements.txt           # Python dependencies
├── setup.py                   # Environment setup script
├── train_model.py            # Main training script
├── predict.py                # Comprehensive prediction script
├── classify_audio.py         # Single file classification
├── evaluate_model.py         # Model evaluation
├── quick_test.py             # Basic functionality test
└── README.md                 # This file
```

## 🛠️ Installation

1. **Clone the repository** (if from git) or ensure you have the project files
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset** in the `Data/` folder with the structure shown above

## 🔧 Model Architecture

### Hybrid Deep Learning Components

1. **Wav2Vec2 Feature Extractor**
   - Pre-trained Wav2Vec2 model for raw audio feature extraction
   - Extracts high-level audio representations

2. **Bidirectional LSTM**
   - Processes sequential features in both directions
   - Captures temporal dependencies in audio signals

3. **Attention Mechanism**
   - Focuses on relevant parts of the audio sequence
   - Improves classification accuracy

4. **CNN for Spectrograms**
   - Convolutional layers for mel-spectrogram processing
   - Extracts spatial-frequency features

5. **Traditional Feature Processing**
   - MFCC (Mel-frequency cepstral coefficients)
   - Spectral features (centroid, rolloff, bandwidth)
   - Rhythm features (tempo, onset strength)

### Feature Fusion
All features are combined through a fusion layer before final classification.

## 📊 Data Processing

### Feature Extraction
- **MFCC**: 13 coefficients + delta + delta-delta
- **Spectral**: Centroid, rolloff, bandwidth, zero-crossing rate
- **Rhythm**: Tempo and onset strength patterns
- **Spectrograms**: Mel-spectrograms for CNN processing
- **Raw Audio**: Direct waveform for Wav2Vec2

### Data Balancing
- Uses SMOTE (Synthetic Minority Oversampling Technique)
- Addresses class imbalance in the dataset
- Original distribution: Hungry (382), Discomfort (135), Tired (132), Belly Pain (124), Cold/Hot (100)

### Data Augmentation
- Gaussian noise addition
- Time stretching (0.8x to 1.25x)
- Pitch shifting (±4 semitones)
- Time shifting (±50% of duration)

## 🎯 Usage

### Training the Model

1. **Preprocess and train**:
   ```bash
   python train_model.py
   ```
   This will:
   - Load and preprocess the audio data
   - Extract multiple features
   - Balance the dataset using SMOTE
   - Train the hybrid model
   - Save the trained model and artifacts

### Running the Web Application

1. **Start the Flask app**:
   ```bash
   python web_app/app.py
   ```

2. **Open your browser** and go to `http://localhost:5000`

3. **Upload an audio file** (WAV, MP3, FLAC, M4A) and get predictions

### Using the Model Programmatically

```python
import torch
import numpy as np
from src.hybrid_model import HybridCryClassifier
from src.data_preprocessing import AudioPreprocessor

# Load trained model
model = HybridCryClassifier(num_classes=5)
model.load_state_dict(torch.load('models/hybrid_model.pth')['model_state_dict'])
model.eval()

# Preprocess audio
preprocessor = AudioPreprocessor('Data')
features = preprocessor.extract_all_features(['path/to/audio.wav'], [0])

# Make prediction
with torch.no_grad():
    output, attention = model(
        torch.FloatTensor(features['raw_audio']),
        torch.FloatTensor(features['spectrograms']),
        torch.FloatTensor(features['mfcc']),
        torch.FloatTensor(features['spectral']),
        torch.FloatTensor(features['rhythm'])
    )
    prediction = torch.argmax(output, dim=1)
```

## 📈 Performance

The model achieves high accuracy through:
- Multi-modal feature fusion
- Attention mechanisms for focus
- Ensemble learning capabilities
- Balanced training data

Expected performance:
- Overall accuracy: ~85-92%
- High precision for distinct cry types
- Balanced recall across all classes

## 🖥️ Web Interface Features

- **Drag & Drop Upload**: Easy file upload interface
- **Real-time Processing**: Instant predictions
- **Visual Results**: Interactive charts and probability bars
- **Responsive Design**: Works on desktop and mobile
- **Multiple Formats**: Supports WAV, MP3, FLAC, M4A

## 🔍 Model Details

### Input Requirements
- **Sample Rate**: 16kHz (automatically resampled)
- **Duration**: 5 seconds (padded or truncated)
- **Format**: Any common audio format

### Output
- **Predicted Class**: Most likely cry type
- **Confidence Score**: Probability of prediction
- **All Probabilities**: Scores for all 5 classes
- **Attention Weights**: Which parts of audio were most important

## ⚠️ Important Notes

- This system is for **educational and research purposes**
- Results should **not replace medical advice**
- Always **consult healthcare professionals** for medical concerns
- Model performance may vary with different recording conditions
- Best results with clear audio and minimal background noise

## 🛠️ Technical Requirements

- Python 3.7+
- PyTorch 2.0+
- 8GB+ RAM (for training)
- GPU recommended but not required
- 2GB+ storage for model and data

## 📝 Dependencies

Key libraries:
- **PyTorch**: Deep learning framework
- **Transformers**: Wav2Vec2 model
- **Librosa**: Audio processing
- **Scikit-learn**: Data preprocessing
- **Flask**: Web framework
- **Imbalanced-learn**: Data balancing

## 🚀 Future Improvements

- Real-time audio streaming
- Mobile app development
- Multi-language support
- Advanced ensemble methods
- Larger dataset training
- Edge device deployment

## 📊 Dataset

The dataset contains baby cry recordings categorized into:
- **Belly Pain**: Digestive discomfort, colic
- **Cold/Hot**: Temperature-related discomfort
- **Discomfort**: General irritation
- **Hungry**: Need for feeding
- **Tired**: Need for sleep

## 🤝 Contributing

Feel free to contribute by:
- Improving model architecture
- Adding more features
- Enhancing the web interface
- Optimizing performance
- Adding documentation

## 📄 License

This project is for educational purposes. Please ensure proper attribution when using.

---
