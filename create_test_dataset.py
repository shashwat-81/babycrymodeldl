#!/usr/bin/env python3
"""
Create a test dataset with sample audio files for easy classification testing
"""

import os
import shutil
import random
from pathlib import Path

def create_test_dataset():
    """Create a curated test dataset with representative samples from each class"""
    
    print("🔧 Creating Test Dataset for Baby Cry Classification")
    print("=" * 60)
    
    # Define source and destination paths
    source_data_dir = Path("Data")
    test_dataset_dir = Path("test_dataset")
    
    # Create test dataset directory
    test_dataset_dir.mkdir(exist_ok=True)
    
    # Define classes and their source directories
    classes = {
        'belly_pain': 'belly pain',
        'cold_hot': 'cold_hot', 
        'discomfort': 'discomfort',
        'hungry': 'hungry',
        'tired': 'tired'
    }
    
    # Number of samples per class for testing
    samples_per_class = 10
    
    total_copied = 0
    
    for class_name, source_folder in classes.items():
        print(f"\n📁 Processing {class_name}...")
        
        # Create class directory in test dataset
        class_dir = test_dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Get source directory
        source_dir = source_data_dir / source_folder
        
        if not source_dir.exists():
            print(f"   ⚠️  Source directory not found: {source_dir}")
            continue
            
        # Get all .wav files
        wav_files = list(source_dir.glob("*.wav"))
        
        if not wav_files:
            print(f"   ⚠️  No WAV files found in {source_dir}")
            continue
            
        print(f"   📊 Found {len(wav_files)} audio files")
        
        # Select representative samples
        selected_files = []
        
        # Try to get diverse samples
        if len(wav_files) >= samples_per_class:
            # Sort files and select evenly distributed samples
            sorted_files = sorted(wav_files)
            step = len(sorted_files) // samples_per_class
            selected_files = [sorted_files[i * step] for i in range(samples_per_class)]
        else:
            # Use all available files if less than desired
            selected_files = wav_files[:samples_per_class]
        
        # Copy selected files to test dataset
        for i, source_file in enumerate(selected_files):
            dest_file = class_dir / f"{class_name}_sample_{i+1:02d}.wav"
            
            try:
                shutil.copy2(source_file, dest_file)
                print(f"   ✅ Copied: {dest_file.name}")
                total_copied += 1
            except Exception as e:
                print(f"   ❌ Error copying {source_file.name}: {e}")
    
    # Create a README file for the test dataset
    readme_content = f"""# Baby Cry Classification Test Dataset

This test dataset contains {total_copied} carefully selected audio samples for testing the baby cry classification model.

## Dataset Structure
```
test_dataset/
├── belly_pain/        # Baby crying due to stomach pain
├── cold_hot/          # Baby crying due to temperature discomfort  
├── discomfort/        # Baby crying due to general discomfort
├── hungry/            # Baby crying due to hunger
└── tired/             # Baby crying due to tiredness
```

## How to Use

### 1. Web Interface Testing
Run the Flask app and upload files from this dataset:
```bash
cd d:\\Baby-Cry-Model
python app.py
```
Then open http://localhost:5000 and upload files from each class folder.

### 2. Batch Testing Script
Use the batch testing script to test all samples:
```bash
python test_batch_classification.py
```

### 3. Single File Testing
Test individual files using the command line:
```bash
python classify_audio.py test_dataset/belly_pain/belly_pain_sample_01.wav
```

## Expected Results
The model should achieve approximately 70% accuracy on this test set, with:
- **Best performance**: belly_pain, cold_hot classes (~80-85% accuracy)
- **Good performance**: discomfort, tired classes (~70-75% accuracy) 
- **Challenging**: hungry class (~25-35% accuracy)

## Sample File Descriptions
Each class contains {samples_per_class} representative samples that demonstrate typical characteristics of that cry type.

*Generated on: {Path().cwd().name} - {total_copied} total samples*
"""

    with open(test_dataset_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create a file listing for easy reference
    file_listing = []
    for class_dir in test_dataset_dir.iterdir():
        if class_dir.is_dir() and class_dir.name != "__pycache__":
            wav_files = list(class_dir.glob("*.wav"))
            file_listing.append(f"\n## {class_dir.name.upper()} ({len(wav_files)} files)")
            for wav_file in sorted(wav_files):
                file_listing.append(f"- {wav_file.name}")
    
    with open(test_dataset_dir / "file_listing.txt", "w") as f:
        f.write("# Baby Cry Test Dataset - File Listing\n")
        f.write("\n".join(file_listing))
    
    print(f"\n🎉 Test dataset created successfully!")
    print(f"   📍 Location: {test_dataset_dir.absolute()}")
    print(f"   📊 Total samples: {total_copied}")
    print(f"   📋 Classes: {len([d for d in test_dataset_dir.iterdir() if d.is_dir()])}")
    print(f"   📄 Documentation: README.md, file_listing.txt")
    
    return test_dataset_dir, total_copied

if __name__ == "__main__":
    test_dir, total = create_test_dataset()
    
    print(f"\n🚀 Ready for testing!")
    print(f"   Use: python test_batch_classification.py")
    print(f"   Or upload files from {test_dir} to the web interface")