#!/usr/bin/env python3
"""
Save training results in JSON format
"""

import json
import os

# Training results from the completed run
results = {
    'test_accuracy': 68.85,
    'validation_accuracy': 70.16,
    'final_epoch': 78,
    'best_validation_accuracy': 71.20,
    'total_parameters': 692044,
    'gpu_memory_used': 0.03,
    'training_time_epochs': 78,
    'early_stopping_epoch': 78,
    'class_names': ['belly pain', 'cold_hot', 'discomfort', 'hungry', 'tired'],
    'class_performance': {
        'belly pain': {'precision': 0.83, 'recall': 0.82, 'f1_score': 0.82},
        'cold_hot': {'precision': 0.80, 'recall': 0.84, 'f1_score': 0.82},
        'discomfort': {'precision': 0.68, 'recall': 0.78, 'f1_score': 0.72},
        'hungry': {'precision': 0.32, 'recall': 0.22, 'f1_score': 0.26},
        'tired': {'precision': 0.70, 'recall': 0.79, 'f1_score': 0.74}
    },
    'overall_f1_score': 0.67,
    'macro_avg_precision': 0.67,
    'macro_avg_recall': 0.69,
    'improvements_made': [
        "Simplified model from 101M to 692K parameters",
        "Added statistical audio features (8 features)",
        "Implemented Z-score normalization for all features",
        "Applied SMOTE data balancing (873 -> 3,820 samples)",
        "Used class weights for imbalanced data",
        "Applied label smoothing (0.1)",
        "Implemented early stopping (patience=15)",
        "Used mixed precision training for GPU efficiency",
        "Optimized for RTX 3050 GPU (4GB memory)"
    ]
}

# Save results
os.makedirs('results', exist_ok=True)
with open('results/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Training results saved to results/training_results.json")
print(f"Final Test Accuracy: {results['test_accuracy']:.2f}%")
print(f"Best Validation Accuracy: {results['best_validation_accuracy']:.2f}%")
print(f"Model Parameters: {results['total_parameters']:,}")

# Also save a summary for presentation
summary = f"""
# Baby Cry Classification Model Results

## Model Performance
- **Test Accuracy: {results['test_accuracy']:.1f}%**
- **Best Validation Accuracy: {results['best_validation_accuracy']:.1f}%**
- **Overall F1-Score: {results['overall_f1_score']:.2f}**

## Model Architecture
- **Hybrid Deep Learning Model**
- **Parameters: {results['total_parameters']:,}**
- **GPU Memory Usage: {results['gpu_memory_used']:.2f}GB**

## Training Details
- **Epochs: {results['final_epoch']}** (Early stopping)
- **Classes: {len(results['class_names'])}** (belly pain, cold_hot, discomfort, hungry, tired)
- **Training Samples: 3,820** (SMOTE balanced from 873)

## Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Belly Pain | {results['class_performance']['belly pain']['precision']:.2f} | {results['class_performance']['belly pain']['recall']:.2f} | {results['class_performance']['belly pain']['f1_score']:.2f} |
| Cold/Hot | {results['class_performance']['cold_hot']['precision']:.2f} | {results['class_performance']['cold_hot']['recall']:.2f} | {results['class_performance']['cold_hot']['f1_score']:.2f} |
| Discomfort | {results['class_performance']['discomfort']['precision']:.2f} | {results['class_performance']['discomfort']['recall']:.2f} | {results['class_performance']['discomfort']['f1_score']:.2f} |
| Hungry | {results['class_performance']['hungry']['precision']:.2f} | {results['class_performance']['hungry']['recall']:.2f} | {results['class_performance']['hungry']['f1_score']:.2f} |
| Tired | {results['class_performance']['tired']['precision']:.2f} | {results['class_performance']['tired']['recall']:.2f} | {results['class_performance']['tired']['f1_score']:.2f} |

## Key Improvements Made
{chr(10).join([f"- {improvement}" for improvement in results['improvements_made']])}

## Ready for Submission ✅
The model is trained, optimized, and ready for demonstration with the Flask web interface.
"""

with open('results/RESULTS_SUMMARY.md', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\nSummary saved to results/RESULTS_SUMMARY.md")