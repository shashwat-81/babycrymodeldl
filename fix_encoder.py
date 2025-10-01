#!/usr/bin/env python3
"""
Recreate label encoder for baby cry classification
"""

import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create label encoder with known classes
label_encoder = LabelEncoder()
classes = ['belly pain', 'cold_hot', 'discomfort', 'hungry', 'tired']
label_encoder.fit(classes)

# Save the label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Label encoder recreated successfully")
print(f"Classes: {label_encoder.classes_}")

# Test loading
with open('models/label_encoder.pkl', 'rb') as f:
    test_encoder = pickle.load(f)
    
print(f"✅ Test loading successful: {test_encoder.classes_}")