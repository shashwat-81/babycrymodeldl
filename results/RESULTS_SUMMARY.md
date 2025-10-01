
# Baby Cry Classification Model Results

## Model Performance
- **Test Accuracy: 68.8%**
- **Best Validation Accuracy: 71.2%**
- **Overall F1-Score: 0.67**

## Model Architecture
- **Hybrid Deep Learning Model**
- **Parameters: 692,044**
- **GPU Memory Usage: 0.03GB**

## Training Details
- **Epochs: 78** (Early stopping)
- **Classes: 5** (belly pain, cold_hot, discomfort, hungry, tired)
- **Training Samples: 3,820** (SMOTE balanced from 873)

## Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Belly Pain | 0.83 | 0.82 | 0.82 |
| Cold/Hot | 0.80 | 0.84 | 0.82 |
| Discomfort | 0.68 | 0.78 | 0.72 |
| Hungry | 0.32 | 0.22 | 0.26 |
| Tired | 0.70 | 0.79 | 0.74 |

## Key Improvements Made
- Simplified model from 101M to 692K parameters
- Added statistical audio features (8 features)
- Implemented Z-score normalization for all features
- Applied SMOTE data balancing (873 -> 3,820 samples)
- Used class weights for imbalanced data
- Applied label smoothing (0.1)
- Implemented early stopping (patience=15)
- Used mixed precision training for GPU efficiency
- Optimized for RTX 3050 GPU (4GB memory)

## Ready for Submission ✅
The model is trained, optimized, and ready for demonstration with the Flask web interface.
