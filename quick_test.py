#!/usr/bin/env python3
"""
Quick test script to validate the improved model works
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

def test_gpu():
    """Test GPU availability"""
    print("=== GPU Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    return torch.cuda.is_available()

def test_model():
    """Test model creation"""
    print("\n=== Model Test ===")
    from hybrid_model import HybridCryClassifier
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCryClassifier(
        num_classes=5,
        mfcc_dim=156,
        spectral_dim=20,
        rhythm_dim=4,
        spectrogram_shape=(128, 313),
        hidden_dim=128,
        lstm_layers=1
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    raw_audio = torch.randn(batch_size, 80000).to(device)
    spectrograms = torch.randn(batch_size, 128, 313).to(device)
    mfcc = torch.randn(batch_size, 156).to(device)
    spectral = torch.randn(batch_size, 20).to(device)
    rhythm = torch.randn(batch_size, 4).to(device)
    
    with torch.no_grad():
        output, _ = model(raw_audio, spectrograms, mfcc, spectral, rhythm)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {output.shape}")
    print(f"Model test passed!")
    return True

def test_data_loading():
    """Test data loading"""
    print("\n=== Data Loading Test ===")
    
    if not os.path.exists('models/processed_data.npz'):
        print("No processed data found. Run preprocessing first.")
        return False
    
    from data_loader import create_augmented_loaders
    
    # Load processed data
    data = np.load('models/processed_data.npz')
    features_dict = {key: data[key] for key in data.files if not key.startswith('norm_')}
    
    print(f"Data shapes:")
    for key, value in features_dict.items():
        print(f"  {key}: {value.shape}")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_augmented_loaders(
        features_dict, batch_size=4, num_workers=0
    )
    
    # Test one batch
    for batch in train_loader:
        raw_audio, spectrograms, mfcc, spectral, rhythm, labels = batch
        print(f"Batch test passed!")
        print(f"  Raw audio: {raw_audio.shape}")
        print(f"  Labels: {labels.shape}")
        break
    
    return True

def main():
    """Run all tests"""
    print("🚀 Quick Test Script for Baby Cry Classification")
    print("=" * 60)
    
    try:
        gpu_available = test_gpu()
        model_works = test_model()
        data_works = test_data_loading()
        
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        print(f"✅ GPU Available: {gpu_available}")
        print(f"✅ Model Working: {model_works}")
        print(f"✅ Data Loading: {data_works}")
        
        if all([model_works, data_works]):
            print("\n🎉 All tests passed! Ready for training!")
            print("\nRun: python train_model.py")
        else:
            print("\n❌ Some tests failed. Check the issues above.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()