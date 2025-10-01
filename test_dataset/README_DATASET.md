# Test Dataset Structure

**Note**: Audio files are not included in the Git repository due to size constraints.

## Expected Structure
```
test_dataset/
├── belly_pain/
│   ├── belly_pain_sample_01.wav
│   ├── belly_pain_sample_02.wav
│   ├── belly_pain_sample_03.wav
│   ├── belly_pain_sample_04.wav
│   └── belly_pain_sample_05.wav
├── cold_hot/
│   ├── cold_hot_sample_01.wav
│   ├── cold_hot_sample_02.wav
│   ├── cold_hot_sample_03.wav
│   ├── cold_hot_sample_04.wav
│   └── cold_hot_sample_05.wav
├── discomfort/
│   ├── discomfort_sample_01.wav
│   ├── discomfort_sample_02.wav
│   ├── discomfort_sample_03.wav
│   ├── discomfort_sample_04.wav
│   └── discomfort_sample_05.wav
├── hungry/
│   ├── hungry_sample_01.wav
│   ├── hungry_sample_02.wav
│   ├── hungry_sample_03.wav
│   ├── hungry_sample_04.wav
│   └── hungry_sample_05.wav
└── tired/
    ├── tired_sample_01.wav
    ├── tired_sample_02.wav
    ├── tired_sample_03.wav
    ├── tired_sample_04.wav
    └── tired_sample_05.wav
```

## To Generate Test Dataset
Run the following command to create the test dataset from your main data:
```bash
python create_test_dataset.py
```

This will create 25 representative audio samples (5 per class) for testing the model.

## Usage
- **Web Interface**: Upload these files via the Flask app
- **Single File Testing**: `python classify_audio.py test_dataset/belly_pain/belly_pain_sample_01.wav`
- **Batch Testing**: `python test_batch_classification.py`