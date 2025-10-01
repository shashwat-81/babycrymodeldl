import torch
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_preprocessing import AudioPreprocessor
from hybrid_model import HybridCryClassifier, EnsembleClassifier, ModelTrainer
from data_loader import create_augmented_loaders

def main():
    # Set device - explicitly use GPU 0 (discrete RTX 3050)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Force GPU 0
        device = torch.device('cuda:0')
        print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Clear GPU cache and optimize memory
        torch.cuda.empty_cache()
        
        # Set memory allocation strategy
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable memory fraction to prevent OOM (use 85% of GPU memory)
        torch.cuda.set_per_process_memory_fraction(0.85, device=0)
        
        # Show GPU memory info
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Check if processed data exists
    if not os.path.exists('models/processed_data.npz'):
        print("Processed data not found. Running preprocessing...")
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor('Data')
        
        # Load and process data
        audio_files, labels_encoded, labels = preprocessor.load_dataset(augment_data=True)
        features_dict = preprocessor.extract_all_features(audio_files, labels_encoded, augment_data=True)
        balanced_features = preprocessor.balance_data(features_dict, strategy='smote')
        preprocessor.save_processed_data(balanced_features)
        
        print("Data preprocessing completed!")
    else:
        print("Loading existing processed data...")
        
    # Load processed data
    data = np.load('models/processed_data.npz')
    features_dict = {key: data[key] for key in data.files if not key.startswith('norm_')}
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(features_dict['labels'])
    class_weights = compute_class_weight(
        'balanced', 
        classes=unique_classes, 
        y=features_dict['labels']
    )
    print(f"Class weights: {dict(zip(unique_classes, class_weights))}")
    
    # Load label encoder
    if os.path.exists('models/label_encoder.pkl'):
        label_encoder = joblib.load('models/label_encoder.pkl')
    else:
        print("Label encoder not found. Please run preprocessing first.")
        return
    
    print(f"Data shapes:")
    print(f"  Raw audio: {features_dict['raw_audio'].shape}")
    print(f"  Spectrograms: {features_dict['spectrograms'].shape}")
    print(f"  MFCC: {features_dict['mfcc'].shape}")
    print(f"  Spectral: {features_dict['spectral'].shape}")
    print(f"  Rhythm: {features_dict['rhythm'].shape}")
    print(f"  Labels: {features_dict['labels'].shape}")
    
    # Create data loaders with smaller batch size for stability
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_augmented_loaders(
        features_dict, 
        test_size=0.2, 
        val_size=0.1, 
        batch_size=8,  # Smaller batch for stability
        num_workers=0,
        random_state=42
    )
    
    # Model parameters
    num_classes = len(label_encoder.classes_)
    mfcc_dim = features_dict['mfcc'].shape[1]
    spectral_dim = features_dict['spectral'].shape[1]
    rhythm_dim = features_dict['rhythm'].shape[1]
    spectrogram_shape = features_dict['spectrograms'].shape[1:]
    
    print(f"Model parameters:")
    print(f"  Number of classes: {num_classes}")
    print(f"  MFCC dimension: {mfcc_dim}")
    print(f"  Spectral dimension: {spectral_dim}")
    print(f"  Rhythm dimension: {rhythm_dim}")
    print(f"  Spectrogram shape: {spectrogram_shape}")
    
    # Create model with simplified architecture
    model = HybridCryClassifier(
        num_classes=num_classes,
        mfcc_dim=mfcc_dim,
        spectral_dim=spectral_dim,
        rhythm_dim=rhythm_dim,
        spectrogram_shape=spectrogram_shape,
        hidden_dim=128,  # Reduced complexity
        lstm_layers=1    # Simplified
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Feature dimensions - MFCC: {mfcc_dim}, Spectral: {spectral_dim}, Rhythm: {rhythm_dim}")
    
    # Create trainer with better learning rate and class weights
    trainer = ModelTrainer(model, device, learning_rate=0.003, class_weights=class_weights)
    
    # Train model with more epochs
    print("Starting training...")
    val_predictions, val_labels = trainer.train(
        train_loader, 
        val_loader, 
        epochs=100,  # Increased epochs for better training
        save_path='models/hybrid_model.pth'
    )
    
    # Evaluate on validation set
    print("\\nValidation Results:")
    report, cm = trainer.evaluate_model(val_predictions, val_labels, label_encoder)
    print(classification_report(val_labels, val_predictions, target_names=label_encoder.classes_))
    
    # Test on test set
    print("\\nEvaluating on test set...")
    model.load_state_dict(torch.load('models/hybrid_model.pth')['model_state_dict'])
    test_loss, test_acc, test_predictions, test_labels = trainer.validate(test_loader)
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\\nTest Results:")
    print(classification_report(test_labels, test_predictions, target_names=label_encoder.classes_))
    
    # Save final evaluation results
    test_report, test_cm = trainer.evaluate_model(test_predictions, test_labels, label_encoder)
    
    # Save evaluation results
    import json
    with open('models/evaluation_results.json', 'w') as f:
        json.dump({
            'test_accuracy': float(test_acc),
            'test_report': test_report,
            'validation_accuracy': float(best_val_acc),
            'class_names': label_encoder.classes_.tolist()
        }, f, indent=2)
    
    print("Training completed! Results saved to models/")

def train_ensemble():
    """Train ensemble model for better performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training ensemble model on device: {device}")
    
    # Load processed data
    data = np.load('models/processed_data.npz')
    features_dict = {key: data[key] for key in data.files}
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_augmented_loaders(
        features_dict, 
        test_size=0.2, 
        val_size=0.1, 
        batch_size=4,  # Even smaller batch for ensemble
        num_workers=0,
        random_state=42
    )
    
    # Model parameters
    num_classes = len(label_encoder.classes_)
    
    # Create ensemble model
    ensemble_model = EnsembleClassifier(num_classes=num_classes, num_models=3)
    
    # Create trainer
    ensemble_trainer = ModelTrainer(ensemble_model, device, learning_rate=0.0005)
    
    # Train ensemble
    print("Training ensemble model...")
    val_predictions, val_labels = ensemble_trainer.train(
        train_loader, 
        val_loader, 
        epochs=30,
        save_path='models/ensemble_model.pth'
    )
    
    # Evaluate ensemble
    print("\\nEnsemble Validation Results:")
    ensemble_trainer.evaluate_model(val_predictions, val_labels, label_encoder)

if __name__ == "__main__":
    # Train single model
    main()
    
    # Uncomment to train ensemble model (requires more memory and time)
    # print("\\n" + "="*50)
    # print("Training Ensemble Model")
    # print("="*50)
    # train_ensemble()