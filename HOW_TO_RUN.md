# 🍼 Baby Cry Classification Web App

## 🚀 Quick Start

**To run the web application:**

```bash
cd d:\Baby-Cry-Model\web_app
python app.py
```

Then open your browser and go to: **http://localhost:5000**

## 📁 Project Structure

```
Baby-Cry-Model/
├── web_app/
│   ├── app.py              ← Main Flask web application  
│   ├── templates/          ← HTML templates
│   ├── static/             ← CSS, JS, images
│   └── uploads/            ← Uploaded audio files
├── src/                    ← Model source code
├── models/                 ← Trained model files
├── test_dataset/           ← Sample test files
└── results/               ← Training results
```

## 🧪 Testing Options

1. **Web Interface**: Upload files via browser
2. **Single File**: `python classify_audio.py test_dataset/belly_pain/belly_pain_sample_01.wav`
3. **Batch Testing**: `python test_batch_classification.py`

## ✅ Ready for Submission!

Your baby cry classification system is complete and working perfectly! 🎯