import torch
import numpy as np
import os
import pickle
import joblib
import sys
import shutil
from pathlib import Path
sys.path.append('src')
from data_preprocessing import extract_all_features
from hybrid_model import HybridCryClassifier


def move_correct_predictions_to_classes():
    """Move correctly predicted files to their respective class folders"""
    print("🔄 Moving Correctly Predicted Files to Their Class Folders")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = HybridCryClassifier()
    checkpoint = torch.load('models/hybrid_model.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Try loading label encoder
    try:
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Loaded label encoder with pickle")
    except:
        try:
            label_encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Loaded label encoder with joblib")
        except:
            print("⚠️ Using default classes")
            class LabelEncoder:
                def __init__(self):
                    self.classes_ = np.array(['belly pain', 'cold_hot', 'discomfort', 'hungry', 'tired'])
            label_encoder = LabelEncoder()
    
    # Load normalization data
    try:
        norm_data = np.load('models/processed_data.npz')
        print("✅ Loaded normalization data")
    except Exception as e:
        print(f"❌ Could not load normalization data: {e}")
        return
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Define source folders (where audio files currently are) - FIXED PATHS
    source_folders = [
        'Data/hungry',
        'Data/tired', 
        'Data/cold_hot',
        'Data/discomfort',
        'Data/belly pain'
    ]
    
    # Create destination folders for correctly predicted files
    dest_base = 'Data/Correctly_Classified'
    dest_folders = {
        'hungry': os.path.join(dest_base, 'hungry'),
        'tired': os.path.join(dest_base, 'tired'),
        'cold_hot': os.path.join(dest_base, 'cold_hot'),
        'discomfort': os.path.join(dest_base, 'discomfort'),
        'belly pain': os.path.join(dest_base, 'belly pain')  # Note: model uses 'belly pain' with space
    }
    
    # Create destination directories
    for folder in dest_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    print(f"\n✅ Created destination folders in: {dest_base}")
    
    # Track statistics
    total_files = 0
    correctly_moved = 0
    incorrectly_predicted = 0
    errors = 0
    
    # Process each source folder
    for source_folder in source_folders:
        if not os.path.exists(source_folder):
            print(f"⚠️ Source folder not found: {source_folder}")
            continue
            
        folder_name = os.path.basename(source_folder)
        print(f"\n{'='*50}")
        print(f"🔍 Processing: {source_folder}")
        print(f"{'='*50}")
        
        # Get all audio files
        audio_files = [f for f in os.listdir(source_folder) if f.endswith(('.wav', '.mp3', '.m4a'))]
        print(f"Found {len(audio_files)} files")
        
        if len(audio_files) == 0:
            continue
            
        # Process each file
        for file_name in audio_files:
            file_path = os.path.join(source_folder, file_name)
            total_files += 1
            
            try:
                # Extract features
                mfcc, spectral, rhythm, spectrogram, raw_audio = extract_all_features(file_path)
                
                # Normalize features
                mfcc = (mfcc - norm_data['norm_mfcc_mean']) / norm_data['norm_mfcc_std']
                spectral = (spectral - norm_data['norm_spectral_mean']) / norm_data['norm_spectral_std']
                rhythm = (rhythm - norm_data['norm_rhythm_mean']) / norm_data['norm_rhythm_std']
                spectrogram = (spectrogram - norm_data['norm_spec_mean']) / norm_data['norm_spec_std']
                
                # Convert to tensors
                raw_audio_tensor = torch.FloatTensor(raw_audio).unsqueeze(0).to(device)
                spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(device)
                mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
                spectral_tensor = torch.FloatTensor(spectral).unsqueeze(0).to(device)
                rhythm_tensor = torch.FloatTensor(rhythm).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs, _ = model(raw_audio_tensor, spectrogram_tensor, mfcc_tensor,
                                     spectral_tensor, rhythm_tensor)
                    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                
                # Get predicted class
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.classes_[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                # Find the destination folder for this predicted class
                if predicted_class in dest_folders:
                    dest_folder = dest_folders[predicted_class]
                    dest_file_path = os.path.join(dest_folder, file_name)
                    
                    # Copy file to the predicted class folder
                    shutil.copy2(file_path, dest_file_path)
                    correctly_moved += 1
                    
                    print(f"✅ {file_name} -> {predicted_class} ({confidence:.2%})")
                else:
                    print(f"❌ Unknown class prediction: {predicted_class}")
                    incorrectly_predicted += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {file_name}: {e}")
                errors += 1
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"🎯 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total files processed: {total_files}")
    print(f"Successfully moved: {correctly_moved}")
    print(f"Incorrectly predicted: {incorrectly_predicted}")
    print(f"Errors: {errors}")
    print(f"Success rate: {(correctly_moved/total_files*100):.2f}%" if total_files > 0 else "0%")
    
    # Show distribution by predicted class
    print(f"\n📊 Distribution by Predicted Class:")
    for class_name, folder_path in dest_folders.items():
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.m4a'))])
            print(f"  {class_name}: {file_count} files")
        else:
            print(f"  {class_name}: 0 files (folder not created)")
    
    # Create summary file
    summary_path = os.path.join(dest_base, 'classification_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Baby Cry Classification Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Successfully classified: {correctly_moved}\n")
        f.write(f"Incorrectly predicted: {incorrectly_predicted}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Success rate: {(correctly_moved/total_files*100):.2f}%\n\n")
        
        f.write("Distribution by Predicted Class:\n")
        f.write("-" * 30 + "\n")
        for class_name, folder_path in dest_folders.items():
            if os.path.exists(folder_path):
                file_count = len([f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.m4a'))])
                f.write(f"{class_name}: {file_count} files\n")
        
        f.write(f"\nFiles organized in: {dest_base}\n")
        f.write("Each subfolder contains files that the model predicted as that class.\n")
    
    print(f"\n✅ Classification complete!")
    print(f"📁 Files organized in: {dest_base}")
    print(f"📄 Summary saved to: {summary_path}")
    print(f"\n💡 Each subfolder now contains files that the model predicts as that class")

def evaluate_correctly_classified_data():
    """Evaluate model performance on correctly classified data"""
    print("🔍 Evaluating Model on Correctly Classified Data")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = HybridCryClassifier()
    checkpoint = torch.load('models/hybrid_model.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Try loading label encoder
    try:
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Loaded label encoder with pickle")
    except:
        try:
            label_encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Loaded label encoder with joblib")
        except:
            print("⚠️ Using default classes")
            class LabelEncoder:
                def __init__(self):
                    self.classes_ = np.array(['belly pain', 'cold_hot', 'discomfort', 'hungry', 'tired'])
            label_encoder = LabelEncoder()
    
    # Load normalization data
    try:
        norm_data = np.load('models/processed_data.npz')
        print("✅ Loaded normalization data")
    except Exception as e:
        print(f"❌ Could not load normalization data: {e}")
        return
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Define correctly classified source folders
    source_base = 'Data/Correctly_Classified'
    
    # Check if correctly classified data exists
    if not os.path.exists(source_base):
        print(f"❌ Correctly classified data not found at: {source_base}")
        print("Please run the organization script first to create correctly classified data.")
        return
    
    # Define class mappings (folder name -> model class name)
    class_mappings = {
        'hungry': 'hungry',
        'tired': 'tired',
        'cold_hot': 'cold_hot', 
        'discomfort': 'discomfort',
        'belly_pain': 'belly pain'  # folder name -> model class name
    }
    
    # Track statistics
    total_files = 0
    correct_predictions = 0
    all_predictions = []
    all_true_labels = []
    class_stats = {}
    
    print(f"\n📊 Processing correctly classified data from: {source_base}")
    
    # Process each class folder
    for folder_name, expected_class in class_mappings.items():
        folder_path = os.path.join(source_base, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue
        
        print(f"\n{'='*50}")
        print(f"🔍 Processing: {folder_name} (expected: {expected_class})")
        print(f"{'='*50}")
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.m4a'))]
        print(f"Found {len(audio_files)} files")
        
        if len(audio_files) == 0:
            class_stats[folder_name] = {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'predictions': {}
            }
            continue
        
        # Initialize class stats
        class_correct = 0
        class_predictions = {}
        
        # Process each file
        for file_name in audio_files:
            file_path = os.path.join(folder_path, file_name)
            total_files += 1
            
            try:
                # Extract features
                mfcc, spectral, rhythm, spectrogram, raw_audio = extract_all_features(file_path)
                
                # Normalize features
                mfcc = (mfcc - norm_data['norm_mfcc_mean']) / norm_data['norm_mfcc_std']
                spectral = (spectral - norm_data['norm_spectral_mean']) / norm_data['norm_spectral_std']
                rhythm = (rhythm - norm_data['norm_rhythm_mean']) / norm_data['norm_rhythm_std']
                spectrogram = (spectrogram - norm_data['norm_spec_mean']) / norm_data['norm_spec_std']
                
                # Convert to tensors
                raw_audio_tensor = torch.FloatTensor(raw_audio).unsqueeze(0).to(device)
                spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(device)
                mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
                spectral_tensor = torch.FloatTensor(spectral).unsqueeze(0).to(device)
                rhythm_tensor = torch.FloatTensor(rhythm).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs, _ = model(raw_audio_tensor, spectrogram_tensor, mfcc_tensor,
                                     spectral_tensor, rhythm_tensor)
                    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                
                # Get predicted class
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.classes_[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                # Track predictions
                if predicted_class not in class_predictions:
                    class_predictions[predicted_class] = 0
                class_predictions[predicted_class] += 1
                
                # Get true label index
                expected_idx = np.where(label_encoder.classes_ == expected_class)[0][0]
                all_predictions.append(predicted_idx)
                all_true_labels.append(expected_idx)
                
                # Check if correct
                if predicted_class == expected_class:
                    class_correct += 1
                    correct_predictions += 1
                    print(f"✅ {file_name}: {predicted_class} ({confidence:.2%})")
                else:
                    print(f"❌ {file_name}: {predicted_class} ({confidence:.2%}) - Expected: {expected_class}")
                
            except Exception as e:
                print(f"⚠️ Error processing {file_name}: {e}")
        
        # Calculate class accuracy
        class_accuracy = (class_correct / len(audio_files) * 100) if len(audio_files) > 0 else 0
        
        class_stats[folder_name] = {
            'total': len(audio_files),
            'correct': class_correct,
            'accuracy': class_accuracy,
            'predictions': class_predictions
        }
        
        print(f"\n📊 {folder_name.upper()} Summary:")
        print(f"  Total files: {len(audio_files)}")
        print(f"  Correct predictions: {class_correct}")
        print(f"  Accuracy: {class_accuracy:.2f}%")
        
        if class_predictions:
            print(f"  Prediction breakdown:")
            for pred_class, count in sorted(class_predictions.items()):
                percentage = (count / len(audio_files)) * 100
                print(f"    {pred_class}: {count} files ({percentage:.1f}%)")
    
    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_files * 100) if total_files > 0 else 0
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"🎯 FINAL SUMMARY - CORRECTLY CLASSIFIED DATA EVALUATION")
    print(f"{'='*70}")
    print(f"Total files processed: {total_files}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_files - correct_predictions}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    
    # Per-class breakdown
    print(f"\n📊 PER-CLASS ACCURACY ON CORRECTLY CLASSIFIED DATA:")
    print(f"{'Class':<15} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    print(f"{'-'*45}")
    for class_name, stats in class_stats.items():
        print(f"{class_name:<15} {stats['correct']:<8} {stats['total']:<8} {stats['accuracy']:<10.2f}%")
    
    # Generate confusion matrix and classification report
    if len(all_predictions) > 0:
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"\n📈 Classification Report:")
        print(classification_report(
            all_true_labels, 
            all_predictions, 
            target_names=label_encoder.classes_,
            zero_division=0
        ))
        
        print(f"\n🔄 Confusion Matrix:")
        cm = confusion_matrix(all_true_labels, all_predictions)
        print("Predicted ->")
        print(f"{'True':<12}", end="")
        for class_name in label_encoder.classes_:
            print(f"{class_name:<12}", end="")
        print()
        
        for i, true_class in enumerate(label_encoder.classes_):
            print(f"{true_class:<12}", end="")
            for j in range(len(label_encoder.classes_)):
                print(f"{cm[i][j]:<12}", end="")
            print()
    
    # Save detailed results
    results_dir = 'Data/Correctly_Classified'
    summary_path = os.path.join(results_dir, 'evaluation_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Correctly Classified Data - Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Statistics:\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Correct predictions: {correct_predictions}\n")
        f.write(f"Incorrect predictions: {total_files - correct_predictions}\n")
        f.write(f"Overall accuracy: {overall_accuracy:.2f}%\n\n")
        
        f.write("Per-Class Results:\n")
        f.write("-" * 30 + "\n")
        for class_name, stats in class_stats.items():
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Total files: {stats['total']}\n")
            f.write(f"  Correct predictions: {stats['correct']}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.2f}%\n")
            
            if stats['predictions']:
                f.write(f"  Prediction breakdown:\n")
                for pred_class, count in sorted(stats['predictions'].items()):
                    percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
                    f.write(f"    {pred_class}: {count} files ({percentage:.1f}%)\n")
        
        if len(all_predictions) > 0:
            f.write(f"\nClassification Report:\n")
            f.write("-" * 30 + "\n")
            f.write(classification_report(
                all_true_labels, 
                all_predictions, 
                target_names=label_encoder.classes_,
                zero_division=0
            ))
    
    print(f"\n✅ Evaluation complete!")
    print(f"📄 Detailed results saved to: {summary_path}")
    
    # Provide insights
    print(f"\n💡 INSIGHTS:")
    
    if overall_accuracy > 90:
        print(f"🎉 Excellent! The model maintains {overall_accuracy:.1f}% accuracy on correctly classified data.")
    elif overall_accuracy > 80:
        print(f"✅ Good! The model has {overall_accuracy:.1f}% accuracy on correctly classified data.")
    elif overall_accuracy > 70:
        print(f"⚠️ Moderate performance: {overall_accuracy:.1f}% accuracy. Some data might need re-labeling.")
    else:
        print(f"❌ Low accuracy: {overall_accuracy:.1f}%. The correctly classified data may have issues.")
    
    # Find best and worst performing classes
    best_class = max(class_stats.items(), key=lambda x: x[1]['accuracy'])
    worst_class = min(class_stats.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"🏆 Best performing class: {best_class[0]} ({best_class[1]['accuracy']:.1f}%)")
    print(f"⚠️ Worst performing class: {worst_class[0]} ({worst_class[1]['accuracy']:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Move correctly predicted files to class folders')
    parser.add_argument('--auto', action='store_true', help='Run without confirmation prompts')
    args = parser.parse_args()
    
    if not args.auto:
        print("This script will:")
        print("1. Test ALL audio files in Data/[class_name] folders")
        print("2. Move each file to Data/Correctly_Classified/[predicted_class]")
        print("3. Generate classification summary")
        
        response = input("\nProceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            return
    
    move_correct_predictions_to_classes()

if __name__ == "__main__":
    main()


