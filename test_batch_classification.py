#!/usr/bin/env python3
"""
Batch testing script for baby cry classification model
Tests all files in the test dataset and provides detailed results
"""

import os
import sys
import torch
import pickle
import librosa
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json

# Add src to path
sys.path.append('src')
from data_preprocessing import extract_all_features
from hybrid_model import HybridCryClassifier

def load_model_and_encoder():
    """Load the trained model and label encoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    return model, label_encoder, device

def predict_audio_file(model, label_encoder, device, audio_path):
    """Predict the class of a single audio file"""
    try:
        # Load normalization parameters
        norm_data = np.load('models/processed_data.npz')
        
        # Extract features from audio file
        mfcc, spectral, rhythm, spectrogram, raw_audio = extract_all_features(str(audio_path))
        
        # Normalize features using training statistics
        mfcc = (mfcc - norm_data['norm_mfcc_mean']) / norm_data['norm_mfcc_std']
        spectral = (spectral - norm_data['norm_spectral_mean']) / norm_data['norm_spectral_std']
        rhythm = (rhythm - norm_data['norm_rhythm_mean']) / norm_data['norm_rhythm_std']
        spectrogram = (spectrogram - norm_data['norm_spec_mean']) / norm_data['norm_spec_std']
        
        # Convert to tensors and add batch dimension
        raw_audio_tensor = torch.FloatTensor(raw_audio).unsqueeze(0).to(device)
        spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(device)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
        spectral_tensor = torch.FloatTensor(spectral).unsqueeze(0).to(device)
        rhythm_tensor = torch.FloatTensor(rhythm).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs, _ = model(raw_audio_tensor, spectrogram_tensor, mfcc_tensor, spectral_tensor, rhythm_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        predicted_class = label_encoder.classes_[predicted_class_idx]
        all_probs = {label_encoder.classes_[i]: probabilities[0][i].item() 
                    for i in range(len(label_encoder.classes_))}
        
        return predicted_class, confidence, all_probs
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, 0.0, {}

def test_batch_classification():
    """Test all files in the test dataset"""
    print("🧪 Batch Testing Baby Cry Classification Model")
    print("=" * 60)
    
    # Check if test dataset exists
    test_dataset_dir = Path("test_dataset")
    if not test_dataset_dir.exists():
        print("❌ Test dataset not found! Please run create_test_dataset.py first.")
        return
    
    # Load model
    print("🔧 Loading model and encoder...")
    try:
        model, label_encoder, device = load_model_and_encoder()
        print(f"✅ Model loaded on {device}")
        print(f"   Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Collect all test files
    test_files = []
    true_labels = []
    
    for class_dir in test_dataset_dir.iterdir():
        if class_dir.is_dir() and class_dir.name in label_encoder.classes_:
            wav_files = list(class_dir.glob("*.wav"))
            for wav_file in wav_files:
                test_files.append(wav_file)
                true_labels.append(class_dir.name)
    
    if not test_files:
        print("❌ No test files found in test dataset!")
        return
    
    print(f"📊 Found {len(test_files)} test files across {len(set(true_labels))} classes")
    
    # Make predictions
    predictions = []
    confidences = []
    all_probabilities = []
    successful_predictions = 0
    
    print("\n🔍 Processing test files...")
    for i, (test_file, true_label) in enumerate(zip(test_files, true_labels)):
        print(f"   ({i+1:2d}/{len(test_files)}) {test_file.name}...", end="")
        
        predicted_class, confidence, all_probs = predict_audio_file(model, label_encoder, device, test_file)
        
        if predicted_class is not None:
            predictions.append(predicted_class)
            confidences.append(confidence)
            all_probabilities.append(all_probs)
            successful_predictions += 1
            
            # Show result
            status = "✅" if predicted_class == true_label else "❌"
            print(f" {status} {predicted_class} ({confidence:.1%})")
        else:
            predictions.append("ERROR")
            confidences.append(0.0)
            all_probabilities.append({})
            print(" ❌ ERROR")
    
    # Calculate metrics
    valid_predictions = [p for p in predictions if p != "ERROR"]
    valid_true_labels = [true_labels[i] for i, p in enumerate(predictions) if p != "ERROR"]
    
    if not valid_predictions:
        print("❌ No successful predictions made!")
        return
    
    accuracy = sum(1 for p, t in zip(valid_predictions, valid_true_labels) if p == t) / len(valid_predictions)
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Total files: {len(test_files)}")
    print(f"   Successful predictions: {successful_predictions}")
    print(f"   Overall accuracy: {accuracy:.1%}")
    print(f"   Average confidence: {np.mean([c for c in confidences if c > 0]):.1%}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(valid_true_labels, valid_predictions, zero_division=0))
    
    # Confusion matrix
    print(f"\n🔄 Confusion Matrix:")
    cm = confusion_matrix(valid_true_labels, valid_predictions, labels=label_encoder.classes_)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df.to_string())
    
    # Per-class results
    print(f"\n🎯 Per-Class Results:")
    class_results = {}
    for class_name in label_encoder.classes_:
        class_true = [i for i, t in enumerate(valid_true_labels) if t == class_name]
        class_pred = [valid_predictions[i] for i in class_true]
        
        if class_true:
            class_accuracy = sum(1 for p in class_pred if p == class_name) / len(class_pred)
            class_confidence = np.mean([confidences[i] for i in class_true if confidences[i] > 0])
            class_results[class_name] = {
                'accuracy': class_accuracy,
                'confidence': class_confidence,
                'samples': len(class_true)
            }
            print(f"   {class_name:12s}: {class_accuracy:.1%} accuracy, {class_confidence:.1%} confidence ({len(class_true)} samples)")
    
    # Save detailed results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    detailed_results = {
        'overall_accuracy': float(accuracy),
        'total_files': len(test_files),
        'successful_predictions': successful_predictions,
        'class_results': {k: {sub_k: float(sub_v) if isinstance(sub_v, (int, float, np.floating)) else sub_v 
                             for sub_k, sub_v in v.items()} for k, v in class_results.items()},
        'file_results': []
    }
    
    # Add individual file results
    for i, (test_file, true_label) in enumerate(zip(test_files, true_labels)):
        if i < len(predictions):
            detailed_results['file_results'].append({
                'filename': test_file.name,
                'true_label': true_label,
                'predicted_label': predictions[i],
                'confidence': float(confidences[i]),
                'correct': predictions[i] == true_label,
                'probabilities': {k: float(v) for k, v in all_probabilities[i].items()} if i < len(all_probabilities) else {}
            })
    
    # Save results
    with open(results_dir / "batch_test_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_dir}/batch_test_results.json")
    print(f"🎉 Batch testing completed!")
    
    return detailed_results

if __name__ == "__main__":
    results = test_batch_classification()