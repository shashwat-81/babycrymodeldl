import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from hybrid_model import HybridCryClassifier, ModelTrainer
from data_loader import create_augmented_loaders

def evaluate_model():
    """Comprehensive model evaluation"""
    print("🔍 Baby Cry Classification - Model Evaluation")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('models/hybrid_model.pth'):
        print("❌ Trained model not found!")
        print("Please run train_model.py first to train the model.")
        return
    
    # Load processed data
    if not os.path.exists('models/processed_data.npz'):
        print("❌ Processed data not found!")
        print("Please run train_model.py first to preprocess data.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading processed data...")
    data = np.load('models/processed_data.npz')
    features_dict = {key: data[key] for key in data.files}
    
    # Load label encoder
    label_encoder = joblib.load('models/label_encoder.pkl')
    class_names = label_encoder.classes_
    
    print(f"Classes: {list(class_names)}")
    print(f"Total samples: {len(features_dict['labels'])}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_augmented_loaders(
        features_dict, 
        test_size=0.2, 
        val_size=0.1, 
        batch_size=16,
        num_workers=0,
        random_state=42
    )
    
    # Model parameters
    num_classes = len(class_names)
    mfcc_dim = features_dict['mfcc'].shape[1]
    spectral_dim = features_dict['spectral'].shape[1]
    rhythm_dim = features_dict['rhythm'].shape[1]
    spectrogram_shape = features_dict['spectrograms'].shape[1:]
    
    # Create and load model
    print("🤖 Loading trained model...")
    model = HybridCryClassifier(
        num_classes=num_classes,
        mfcc_dim=mfcc_dim,
        spectral_dim=spectral_dim,
        rhythm_dim=rhythm_dim,
        spectrogram_shape=spectrogram_shape,
        hidden_dim=256,
        lstm_layers=2
    )
    
    checkpoint = torch.load('models/hybrid_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    
    # Create trainer for evaluation
    trainer = ModelTrainer(model, device)
    
    # Evaluate on test set
    print("🔬 Evaluating on test set...")
    test_loss, test_acc, test_predictions, test_labels = trainer.validate(test_loader)
    
    print(f"📊 Test Results:")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Loss: {test_loss:.4f}")
    
    # Detailed classification report
    print("\\n📋 Detailed Classification Report:")
    report = classification_report(test_labels, test_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    print(classification_report(test_labels, test_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy
    print("\\n📊 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        class_acc = report[class_name]['f1-score']
        print(f"{class_name}: F1-Score = {class_acc:.3f}")
    
    # Class distribution in test set
    unique, counts = np.unique(test_labels, return_counts=True)
    print("\\n📈 Test Set Distribution:")
    for label, count in zip(unique, counts):
        class_name = class_names[label]
        percentage = (count / len(test_labels)) * 100
        print(f"{class_name}: {count} samples ({percentage:.1f}%)")
    
    # Validation set evaluation
    print("\\n🔬 Evaluating on validation set...")
    val_loss, val_acc, val_predictions, val_labels = trainer.validate(val_loader)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    # Save detailed results
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'validation_accuracy': float(val_acc),
        'validation_loss': float(val_loss),
        'classification_report': report,
        'class_names': class_names.tolist(),
        'confusion_matrix': cm.tolist(),
        'model_parameters': {
            'num_classes': num_classes,
            'mfcc_dim': mfcc_dim,
            'spectral_dim': spectral_dim,
            'rhythm_dim': rhythm_dim,
            'spectrogram_shape': list(spectrogram_shape)
        }
    }
    
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\n💾 Results saved to 'models/evaluation_results.json'")
    print("📊 Confusion matrix saved to 'models/confusion_matrix_test.png'")
    
    # Model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\\n🔧 Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Error analysis
    print("\\n🔍 Error Analysis:")
    errors = np.where(np.array(test_predictions) != np.array(test_labels))[0]
    print(f"Total errors: {len(errors)} out of {len(test_labels)} ({len(errors)/len(test_labels)*100:.1f}%)")
    
    if len(errors) > 0:
        print("Most confused classes:")
        error_pairs = {}
        for error_idx in errors:
            true_class = class_names[test_labels[error_idx]]
            pred_class = class_names[test_predictions[error_idx]]
            pair = f"{true_class} → {pred_class}"
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_errors[:5]:
            print(f"  {pair}: {count} times")
    
    print("\\n✅ Evaluation completed!")

if __name__ == "__main__":
    evaluate_model()