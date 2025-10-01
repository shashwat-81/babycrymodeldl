#!/usr/bin/env python3
"""
Single file classification script for testing individual audio files
"""

import sys
import torch
import pickle
import librosa
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.append('src')
from data_preprocessing import extract_all_features
from hybrid_model import HybridCryClassifier

def classify_single_audio(audio_path):
    """Classify a single audio file"""
    print(f"🎵 Classifying: {Path(audio_path).name}")
    print("=" * 50)
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f"❌ File not found: {audio_path}")
        return None
    
    # Load model and encoder
    print("🔧 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load model
        model = HybridCryClassifier()
        checkpoint = torch.load('models/hybrid_model.pth', map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Extract features
    print("🔍 Extracting features...")
    try:
        # Load normalization parameters
        norm_data = np.load('models/processed_data.npz')
        
        # Extract features
        mfcc, spectral, rhythm, spectrogram, raw_audio = extract_all_features(audio_path)
        
        # Normalize features
        mfcc = (mfcc - norm_data['norm_mfcc_mean']) / norm_data['norm_mfcc_std']
        spectral = (spectral - norm_data['norm_spectral_mean']) / norm_data['norm_spectral_std']
        rhythm = (rhythm - norm_data['norm_rhythm_mean']) / norm_data['norm_rhythm_std']
        spectrogram = (spectrogram - norm_data['norm_spec_mean']) / norm_data['norm_spec_std']
        
        print("✅ Features extracted and normalized")
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None
    
    # Make prediction
    print("🧠 Making prediction...")
    try:
        # Convert to tensors
        raw_audio_tensor = torch.FloatTensor(raw_audio).unsqueeze(0).to(device)
        spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(device)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
        spectral_tensor = torch.FloatTensor(spectral).unsqueeze(0).to(device)
        rhythm_tensor = torch.FloatTensor(rhythm).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs, _ = model(raw_audio_tensor, spectrogram_tensor, mfcc_tensor, spectral_tensor, rhythm_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        predicted_class = label_encoder.classes_[predicted_class_idx]
        
        print("✅ Prediction completed")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None
    
    # Display results
    print(f"\n🎯 Classification Results:")
    print(f"   📄 File: {Path(audio_path).name}")
    print(f"   🎭 Predicted Class: {predicted_class}")
    print(f"   🎯 Confidence: {confidence:.1%}")
    
    print(f"\n📊 All Class Probabilities:")
    for i, class_name in enumerate(label_encoder.classes_):
        prob = probabilities[0][i].item()
        bar = "█" * int(prob * 20)  # Visual bar
        print(f"   {class_name:12s}: {prob:6.1%} {bar}")
    
    # Audio file info
    try:
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        print(f"\n📻 Audio Information:")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample Rate: {sr} Hz")
        print(f"   Samples: {len(y):,}")
    except:
        pass
    
    result = {
        'filename': Path(audio_path).name,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {label_encoder.classes_[i]: float(probabilities[0][i].item()) 
                         for i in range(len(label_encoder.classes_))}
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Classify baby cry audio files")
    parser.add_argument("audio_file", help="Path to the audio file (.wav)")
    parser.add_argument("--save", help="Save results to JSON file", action="store_true")
    
    args = parser.parse_args()
    
    result = classify_single_audio(args.audio_file)
    
    if result and args.save:
        import json
        from datetime import datetime
        
        output_file = f"classification_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_audio.py <audio_file.wav> [--save]")
        print("\nExample:")
        print("  python classify_audio.py test_dataset/belly_pain/belly_pain_sample_01.wav")
        print("  python classify_audio.py my_baby_cry.wav --save")
    else:
        main()