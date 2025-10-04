import torch
import numpy as np
import joblib
import sys
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

try:
    from hybrid_model import HybridCryClassifier
    from data_preprocessing import AudioPreprocessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the src directory contains the required modules.")
    sys.exit(1)

class CryPredictor:
    def __init__(self, model_path='models/hybrid_model.pth', 
                 label_encoder_path='models/label_encoder.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        self.preprocessor = None
        
        self.load_model(model_path, label_encoder_path)
    
    def load_model(self, model_path, label_encoder_path):
        """Load the trained model and preprocessor"""
        try:
            # Load label encoder
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                print(f"✅ Label encoder loaded: {list(self.label_encoder.classes_)}")
            else:
                raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
            
            # Initialize preprocessor
            self.preprocessor = AudioPreprocessor('Data')
            
            # Load model
            if os.path.exists(model_path):
                # Create model instance
                num_classes = len(self.label_encoder.classes_)
                self.model = HybridCryClassifier(
                    num_classes=num_classes,
                    mfcc_dim=156,  # Standard MFCC dimension
                    spectral_dim=20,  # Updated spectral dimension
                    rhythm_dim=4,   # Standard rhythm dimension
                    spectrogram_shape=(128, 313),  # Standard spectrogram shape
                    hidden_dim=128,  # Reduced for memory efficiency
                    lstm_layers=1   # Reduced for memory efficiency
                )
                
                # Load trained weights
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                print(f"✅ Model loaded successfully on {self.device}")
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single_audio(self, audio_path):
        """Predict cry type for a single audio file"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Preprocess audio
            print(f"🔄 Processing audio: {audio_path}")
            
            # Load and preprocess audio
            audio = self.preprocessor.load_audio_file(audio_path)
            if audio is None:
                raise ValueError("Failed to load audio file")
            
            # Extract features
            mfcc_feat = self.preprocessor.extract_mfcc_features(audio)
            spectral_feat = self.preprocessor.extract_spectral_features(audio)
            rhythm_feat = self.preprocessor.extract_rhythm_features(audio)
            spectrogram = self.preprocessor.generate_spectrogram(audio)
            
            # Convert to tensors
            with torch.no_grad():
                raw_audio = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
                spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
                mfcc = torch.FloatTensor(mfcc_feat).unsqueeze(0).to(self.device)
                spectral = torch.FloatTensor(spectral_feat).unsqueeze(0).to(self.device)
                rhythm = torch.FloatTensor(rhythm_feat).unsqueeze(0).to(self.device)
                
                # Make prediction
                outputs, attention_weights = self.model(
                    raw_audio, spectrogram_tensor, mfcc, spectral, rhythm
                )
                
                # Get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                # Get class name
                predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                
                # Get all class probabilities
                class_probs = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    class_probs[class_name] = probabilities[0][i].item()
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': class_probs,
                    'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None
                }
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            raise
    
    def predict_batch(self, audio_folder):
        """Predict cry types for all audio files in a folder"""
        audio_folder = Path(audio_folder)
        if not audio_folder.exists():
            raise FileNotFoundError(f"Folder not found: {audio_folder}")
        
        # Find audio files (search recursively through subfolders)
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_folder.rglob(f'*{ext}'))
            audio_files.extend(audio_folder.rglob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_folder}")
            return []
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        results = []
        for audio_file in audio_files:
            try:
                result = self.predict_single_audio(str(audio_file))
                result['filename'] = audio_file.name
                results.append(result)
                
                print(f"✅ {audio_file.name}: {result['predicted_class']} ({result['confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ Error processing {audio_file.name}: {e}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Baby Cry Classification Predictor')
    parser.add_argument('input', help='Path to audio file or folder')
    parser.add_argument('--model', default='models/hybrid_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--encoder', default='models/label_encoder.pkl',
                       help='Path to label encoder')
    parser.add_argument('--output', help='Output file for results (optional)')
    
    args = parser.parse_args()
    
    print("🤖 Baby Cry Classification Predictor")
    print("=" * 40)
    
    try:
        # Initialize predictor
        predictor = CryPredictor(args.model, args.encoder)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single file prediction
            result = predictor.predict_single_audio(str(input_path))
            
            print("\\n📊 Prediction Results:")
            print(f"File: {input_path.name}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("\\nAll Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
            
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\\n💾 Results saved to {args.output}")
        
        elif input_path.is_dir():
            # Batch prediction
            results = predictor.predict_batch(str(input_path))
            
            if results:
                print("\\n📊 Batch Prediction Summary:")
                class_counts = {}
                for result in results:
                    pred_class = result['predicted_class']
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                for class_name, count in class_counts.items():
                    print(f"{class_name}: {count} files")
                
                if args.output:
                    import json
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"\\n💾 Results saved to {args.output}")
        
        else:
            print(f"❌ Input path not found: {input_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())