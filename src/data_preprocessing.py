import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, data_path, sample_rate=16000, duration=5.0):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_length = int(sample_rate * duration)
        self.label_encoder = LabelEncoder()
        
        # Audio augmentation pipeline
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5),
        ])
    
    def load_audio_file(self, file_path):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Pad or truncate to fixed length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine MFCC, delta, and delta-delta
            features = np.concatenate([mfccs, mfccs_delta, mfccs_delta2], axis=0)
            
            # Statistical features
            features_mean = np.mean(features, axis=1)
            features_std = np.std(features, axis=1)
            features_max = np.max(features, axis=1)
            features_min = np.min(features, axis=1)
            
            result = np.concatenate([features_mean, features_std, features_max, features_min])
            return result.astype(np.float32)
        except Exception as e:
            print(f"Warning: MFCC feature extraction failed, using defaults: {e}")
            # Return default MFCC features (13*4 = 52 features for each of 3 types = 156 total)
            return np.zeros(156, dtype=np.float32)
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
            
            # Combine all features and compute statistics
            features = []
            for feature in [spectral_centroids, spectral_rolloff, spectral_bandwidth, zcr]:
                features.extend([
                    float(np.mean(feature)), 
                    float(np.std(feature)), 
                    float(np.max(feature)), 
                    float(np.min(feature))
                ])
            
            # Add chroma and tonnetz statistics
            for feature in [chroma, tonnetz]:
                features.extend([
                    float(np.mean(feature)), 
                    float(np.std(feature))
                ])
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Spectral feature extraction failed, using defaults: {e}")
            # Return default spectral features (4*4 + 2*2 = 20 features)
            return np.zeros(20, dtype=np.float32)
    
    def extract_rhythm_features(self, audio):
        """Extract rhythm-based features"""
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Ensure tempo is a scalar
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item()) if tempo.size > 0 else 120.0
            elif tempo is None or np.isnan(tempo):
                tempo = 120.0  # Default tempo
            
            # Onset strength
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            
            # Rhythm patterns
            rhythm_features = [
                float(tempo),
                float(np.mean(onset_env)),
                float(np.std(onset_env)),
                float(np.var(onset_env))
            ]
            
            return np.array(rhythm_features, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Rhythm feature extraction failed, using defaults: {e}")
            # Return default values if extraction fails
            return np.array([120.0, 0.1, 0.05, 0.0025], dtype=np.float32)
    
    def generate_spectrogram(self, audio):
        """Generate mel-spectrogram"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate, 
                n_mels=128, 
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db.astype(np.float32)
        except Exception as e:
            print(f"Warning: Spectrogram generation failed, using default: {e}")
            # Return default spectrogram shape
            return np.zeros((128, 313), dtype=np.float32)
    
    def load_dataset(self, augment_data=True):
        """Load and preprocess the entire dataset"""
        audio_files = []
        labels = []
        
        # Get class names
        classes = [d for d in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, d))]
        
        print(f"Found classes: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(self.data_path, class_name)
            audio_files_in_class = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            print(f"Processing {len(audio_files_in_class)} files in class '{class_name}'")
            
            for audio_file in audio_files_in_class:
                file_path = os.path.join(class_path, audio_file)
                audio_files.append(file_path)
                labels.append(class_name)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        print(f"Total files: {len(audio_files)}")
        print(f"Class distribution: {pd.Series(labels).value_counts()}")
        
        return audio_files, labels_encoded, labels
    
    def extract_all_features(self, audio_files, labels_encoded, augment_data=True):
        """Extract all features from audio files"""
        mfcc_features = []
        spectral_features = []
        rhythm_features = []
        spectrograms = []
        raw_audio = []
        final_labels = []
        
        for i, (file_path, label) in enumerate(zip(audio_files, labels_encoded)):
            if i % 50 == 0:
                print(f"Processing file {i+1}/{len(audio_files)}")
            
            # Load audio
            audio = self.load_audio_file(file_path)
            if audio is None:
                continue
            
            # Extract features
            mfcc_feat = self.extract_mfcc_features(audio)
            spectral_feat = self.extract_spectral_features(audio)
            rhythm_feat = self.extract_rhythm_features(audio)
            spectrogram = self.generate_spectrogram(audio)
            
            # Store original data
            mfcc_features.append(mfcc_feat)
            spectral_features.append(spectral_feat)
            rhythm_features.append(rhythm_feat)
            spectrograms.append(spectrogram)
            raw_audio.append(audio)
            final_labels.append(label)
            
            # Data augmentation
            if augment_data:
                try:
                    augmented_audio = self.augment(samples=audio, sample_rate=self.sample_rate)
                    
                    # Extract features from augmented audio
                    aug_mfcc = self.extract_mfcc_features(augmented_audio)
                    aug_spectral = self.extract_spectral_features(augmented_audio)
                    aug_rhythm = self.extract_rhythm_features(augmented_audio)
                    aug_spectrogram = self.generate_spectrogram(augmented_audio)
                    
                    mfcc_features.append(aug_mfcc)
                    spectral_features.append(aug_spectral)
                    rhythm_features.append(aug_rhythm)
                    spectrograms.append(aug_spectrogram)
                    raw_audio.append(augmented_audio)
                    final_labels.append(label)
                except Exception as e:
                    print(f"Augmentation failed for {file_path}: {e}")
        
        return {
            'mfcc': np.array(mfcc_features),
            'spectral': np.array(spectral_features),
            'rhythm': np.array(rhythm_features),
            'spectrograms': np.array(spectrograms),
            'raw_audio': np.array(raw_audio),
            'labels': np.array(final_labels)
        }
    
    def balance_data(self, features_dict, strategy='smote'):
        """Balance the dataset using various techniques"""
        print("Original class distribution:")
        unique, counts = np.unique(features_dict['labels'], return_counts=True)
        for i, (label, count) in enumerate(zip(unique, counts)):
            class_name = self.label_encoder.inverse_transform([label])[0]
            print(f"{class_name}: {count}")
        
        if strategy == 'smote':
            # Combine all traditional features for SMOTE
            combined_features = np.concatenate([
                features_dict['mfcc'],
                features_dict['spectral'],
                features_dict['rhythm']
            ], axis=1)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(combined_features, features_dict['labels'])
            
            # Split back the features
            mfcc_dim = features_dict['mfcc'].shape[1]
            spectral_dim = features_dict['spectral'].shape[1]
            
            features_dict['mfcc'] = X_resampled[:, :mfcc_dim]
            features_dict['spectral'] = X_resampled[:, mfcc_dim:mfcc_dim+spectral_dim]
            features_dict['rhythm'] = X_resampled[:, mfcc_dim+spectral_dim:]
            
            # For spectrograms and raw audio, we need to resample by creating indices
            original_labels = features_dict['labels']
            new_size = len(y_resampled)
            
            # Create mapping for resampling spectrograms and raw audio
            resampled_indices = []
            for new_label in y_resampled:
                # Find original samples with this label
                matching_indices = np.where(original_labels == new_label)[0]
                if len(matching_indices) > 0:
                    # Randomly select one of the matching samples
                    selected_idx = np.random.choice(matching_indices)
                    resampled_indices.append(selected_idx)
                else:
                    # Fallback to first sample if no match found
                    resampled_indices.append(0)
            
            # Resample spectrograms and raw audio using the indices
            features_dict['spectrograms'] = features_dict['spectrograms'][resampled_indices]
            features_dict['raw_audio'] = features_dict['raw_audio'][resampled_indices]
            features_dict['labels'] = y_resampled
            
            print("Note: Spectrograms and raw audio are resampled using nearest neighbor mapping")
        
        print("\\nBalanced class distribution:")
        unique, counts = np.unique(features_dict['labels'], return_counts=True)
        for i, (label, count) in enumerate(zip(unique, counts)):
            class_name = self.label_encoder.inverse_transform([label])[0]
            print(f"{class_name}: {count}")
        
        return features_dict
    
    def save_processed_data(self, features_dict, filename='processed_data.npz'):
        """Save processed features to file with normalization"""
        
        # Add feature normalization for better training
        print("Normalizing features...")
        
        # Create a copy to avoid modifying original
        normalized_features = features_dict.copy()
        
        # Normalize MFCC features
        mfcc_mean = np.mean(normalized_features['mfcc'], axis=0, keepdims=True)
        mfcc_std = np.std(normalized_features['mfcc'], axis=0, keepdims=True) + 1e-8
        normalized_features['mfcc'] = (normalized_features['mfcc'] - mfcc_mean) / mfcc_std
        
        # Normalize spectral features
        spectral_mean = np.mean(normalized_features['spectral'], axis=0, keepdims=True)
        spectral_std = np.std(normalized_features['spectral'], axis=0, keepdims=True) + 1e-8
        normalized_features['spectral'] = (normalized_features['spectral'] - spectral_mean) / spectral_std
        
        # Normalize rhythm features
        rhythm_mean = np.mean(normalized_features['rhythm'], axis=0, keepdims=True)
        rhythm_std = np.std(normalized_features['rhythm'], axis=0, keepdims=True) + 1e-8
        normalized_features['rhythm'] = (normalized_features['rhythm'] - rhythm_mean) / rhythm_std
        
        # Normalize spectrograms
        spec_mean = np.mean(normalized_features['spectrograms'])
        spec_std = np.std(normalized_features['spectrograms']) + 1e-8
        normalized_features['spectrograms'] = (normalized_features['spectrograms'] - spec_mean) / spec_std
        
        # Store normalization parameters
        normalization_params = {
            'mfcc_mean': mfcc_mean.flatten(), 'mfcc_std': mfcc_std.flatten(),
            'spectral_mean': spectral_mean.flatten(), 'spectral_std': spectral_std.flatten(),
            'rhythm_mean': rhythm_mean.flatten(), 'rhythm_std': rhythm_std.flatten(),
            'spec_mean': spec_mean, 'spec_std': spec_std
        }
        
        # Combine features and normalization params
        all_data = {**normalized_features, **{f'norm_{k}': v for k, v in normalization_params.items()}}
        
        np.savez_compressed(f'models/{filename}', **all_data)
        print(f"Processed data with normalization saved to models/{filename}")
        print("Feature normalization completed.")
    
    def load_processed_data(self, filename='processed_data.npz'):
        """Load processed features from file"""
        data = np.load(f'models/{filename}')
        return {key: data[key] for key in data.files}
    
    def visualize_data_distribution(self, labels):
        """Visualize class distribution"""
        plt.figure(figsize=(10, 6))
        
        # Convert numeric labels back to class names
        class_names = self.label_encoder.inverse_transform(labels)
        
        # Plot distribution
        sns.countplot(data=pd.DataFrame({'Class': class_names}), x='Class')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('web_app/static/class_distribution.png')
        plt.show()

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = AudioPreprocessor('Data')
    
    # Load dataset
    audio_files, labels_encoded, labels = preprocessor.load_dataset(augment_data=True)
    
    # Extract features
    features_dict = preprocessor.extract_all_features(audio_files, labels_encoded, augment_data=True)
    
    # Balance data
    balanced_features = preprocessor.balance_data(features_dict, strategy='smote')
    
    # Save processed data
    preprocessor.save_processed_data(balanced_features)
    
    # Visualize distribution
    preprocessor.visualize_data_distribution(balanced_features['labels'])

def extract_all_features(audio_path):
    """
    Standalone function to extract all features from a single audio file
    Returns: mfcc, spectral, rhythm, spectrogram, raw_audio
    """
    preprocessor = AudioPreprocessor(data_path="Data")  # Dummy path for initialization
    
    # Load audio
    audio = preprocessor.load_audio_file(audio_path)
    if audio is None:
        raise ValueError(f"Could not load audio file: {audio_path}")
    
    # Extract all features
    mfcc_feat = preprocessor.extract_mfcc_features(audio)
    spectral_feat = preprocessor.extract_spectral_features(audio)
    rhythm_feat = preprocessor.extract_rhythm_features(audio)
    
    # Create spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=preprocessor.sample_rate,
        n_mels=128, 
        fmax=8000
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Raw audio (first 32000 samples)
    raw_audio = audio[:32000] if len(audio) >= 32000 else np.pad(audio, (0, 32000 - len(audio)))
    
    return mfcc_feat, spectral_feat, rhythm_feat, spectrogram, raw_audio