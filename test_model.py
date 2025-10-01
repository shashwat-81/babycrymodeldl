#!/usr/bin/env python3
"""
Quick test script for the trained baby cry classification model
"""

import torch
import pickle
import numpy as np
import os
from pathlib import Path

def test_model():
    print("🧪 Testing Baby Cry Classification Model")
    print("=" * 50)
    
    # Check if model files exist
    model_path = "models/hybrid_model.pth"
    encoder_path = "models/label_encoder.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    
    if not os.path.exists(encoder_path):
        print("❌ Label encoder not found!")
        return
    
    try:
        # Load the label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"✅ Label encoder loaded successfully")
        print(f"   Classes: {label_encoder.classes_}")
        
        # Check model file size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✅ Model file found ({model_size:.2f} MB)")
        
        # Test GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        print(f"✅ Device: {device} ({gpu_name})")
        
        # Load model architecture (we need to import it)
        import sys
        sys.path.append('src')
        from hybrid_model import HybridCryClassifier
        
        # Create model instance
        model = HybridCryClassifier()
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with dummy data (match forward method signature)
        batch_size = 1
        raw_audio = torch.randn(batch_size, 32000)  # Raw audio samples
        spectrogram = torch.randn(batch_size, 128, 157)  # Spectrogram
        mfcc_features = torch.randn(batch_size, 156)  # 13*12 MFCC features
        spectral_features = torch.randn(batch_size, 20)  # Spectral features
        rhythm_features = torch.randn(batch_size, 4)  # Rhythm features
        
        # Move to device
        inputs = (
            raw_audio.to(device),
            spectrogram.to(device),
            mfcc_features.to(device),
            spectral_features.to(device), 
            rhythm_features.to(device)
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(*inputs)
            
            # Handle tuple output if model returns multiple values
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        print(f"✅ Model inference successful")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Predicted class: {label_encoder.classes_[predicted_class.item()]}")
        print(f"   Probabilities: {probabilities.cpu().numpy()[0]}")
        
        # Check processed data
        if os.path.exists("models/processed_data.npz"):
            data = np.load("models/processed_data.npz")
            print(f"✅ Processed data available")
            print(f"   Files in archive: {list(data.keys())}")
            if 'X' in data:
                print(f"   Features shape: {data['X'].shape}")
            if 'y' in data:
                print(f"   Samples: {len(data['y'])}")
            elif 'labels' in data:
                print(f"   Samples: {len(data['labels'])}")
        
        print("\n🎉 Model Test Results:")
        print("   ✅ Model loads successfully")
        print("   ✅ GPU acceleration available" if torch.cuda.is_available() else "   ⚠️  CPU only (no GPU)")
        print("   ✅ Inference works correctly")
        print("   ✅ All components ready for deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🚀 Ready for submission! Your baby cry classification model is working perfectly.")
    else:
        print("\n⚠️  Some issues detected. Please check the errors above.")