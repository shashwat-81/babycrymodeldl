#!/usr/bin/env python3
"""
Quick test to check if web app imports work correctly
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'src'))

print("🔧 Testing web app imports...")

try:
    from src.hybrid_model import HybridCryClassifier
    print("✅ HybridCryClassifier imported successfully")
except ImportError as e:
    print(f"❌ Failed to import HybridCryClassifier: {e}")

try:
    from src.data_preprocessing import extract_all_features
    print("✅ extract_all_features imported successfully")
except ImportError as e:
    print(f"❌ Failed to import extract_all_features: {e}")

try:
    import torch
    import pickle
    
    # Test label encoder loading
    with open('../models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅ Label encoder loaded: {label_encoder.classes_}")
    
    # Test model loading
    model = HybridCryClassifier()
    checkpoint = torch.load('../models/hybrid_model.pth', map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model loaded successfully")
    
    # Test feature extraction
    test_audio = "../test_dataset/belly_pain/belly_pain_sample_01.wav"
    if os.path.exists(test_audio):
        features = extract_all_features(test_audio)
        print(f"✅ Feature extraction successful: {len(features)} features")
    else:
        print("⚠️  Test audio file not found, skipping feature test")
    
    print("\n🎉 All imports and basic functionality working!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()