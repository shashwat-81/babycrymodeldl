# Baby Cry Classification Test Dataset

This test dataset contains 25 carefully selected audio samples for testing the baby cry classification model.

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
cd d:\Baby-Cry-Model
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
Each class contains 5 representative samples that demonstrate typical characteristics of that cry type.

*Generated on: Baby-Cry-Model - 25 total samples*
