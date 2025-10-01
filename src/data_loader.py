import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class DataAugmentation:
    """Improved data augmentation for audio data"""
    
    def __init__(self, noise_prob=0.2, shift_prob=0.3, stretch_prob=0.2):
        self.noise_prob = noise_prob
        self.shift_prob = shift_prob
        self.stretch_prob = stretch_prob
    
    def __call__(self, audio):
        """Apply carefully tuned augmentations"""
        # Apply only one augmentation per sample to avoid over-distortion
        aug_choice = np.random.choice(['none', 'noise', 'shift', 'stretch'], 
                                    p=[0.3, 0.3, 0.3, 0.1])
        
        if aug_choice == 'noise' and np.random.random() < self.noise_prob:
            # Very light noise
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = torch.randn_like(audio) * noise_factor
            audio = audio + noise
        elif aug_choice == 'shift' and np.random.random() < self.shift_prob:
            # Light time shift
            shift_limit = 0.05  # Reduced from 0.2
            audio = AudioTransforms.time_shift(audio, shift_limit)
        elif aug_choice == 'stretch' and np.random.random() < self.stretch_prob:
            # Very light time stretch
            rate_range = (0.95, 1.05)  # Much more conservative
            audio = AudioTransforms.time_stretch(audio, rate_range)
        
        return audio

class AudioTransforms:
    def __init__(self, raw_audio, spectrograms, mfcc_features, spectral_features, 
                 rhythm_features, labels, transform=None):
        self.raw_audio = raw_audio
        self.spectrograms = spectrograms
        self.mfcc_features = mfcc_features
        self.spectral_features = spectral_features
        self.rhythm_features = rhythm_features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get data for this index
        try:
            raw_audio = torch.FloatTensor(self.raw_audio[idx])
            spectrogram = torch.FloatTensor(self.spectrograms[idx])
            mfcc = torch.FloatTensor(self.mfcc_features[idx])
            spectral = torch.FloatTensor(self.spectral_features[idx])
            rhythm = torch.FloatTensor(self.rhythm_features[idx])
            label = torch.LongTensor([self.labels[idx]])[0]
            
            # Validate tensor shapes
            if raw_audio.numel() == 0 or raw_audio.shape[0] == 0:
                print(f"Warning: Invalid raw_audio at index {idx}, shape: {raw_audio.shape}")
                # Return a valid tensor with zeros
                raw_audio = torch.zeros(80000)
            
            # Apply transforms if any
            if self.transform:
                raw_audio = self.transform(raw_audio)
                # Validate after transform
                if raw_audio.numel() == 0 or raw_audio.shape[0] == 0:
                    print(f"Warning: Transform made audio invalid at index {idx}")
                    raw_audio = torch.zeros(80000)
                
            return raw_audio, spectrogram, mfcc, spectral, rhythm, label
            
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return default values
            return (torch.zeros(80000), torch.zeros(128, 157), 
                   torch.zeros(156), torch.zeros(20), torch.zeros(4), 
                   torch.LongTensor([0])[0])

class AudioTransforms:
    """Audio data augmentation transforms"""
    
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        """Add Gaussian noise to audio"""
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
    
    @staticmethod
    def time_shift(audio, shift_limit=0.2):
        """Time shift the audio"""
        if len(audio) == 0:
            return audio
            
        shift = int(np.random.uniform(-shift_limit, shift_limit) * len(audio))
        if shift > 0:
            if shift >= len(audio):
                return torch.zeros_like(audio)
            audio = torch.cat([audio[shift:], torch.zeros(shift)])
        elif shift < 0:
            if -shift >= len(audio):
                return torch.zeros_like(audio)
            audio = torch.cat([torch.zeros(-shift), audio[:shift]])
        return audio
    
    @staticmethod
    def time_stretch(audio, rate_range=(0.8, 1.2)):
        """Time stretch the audio (simple implementation)"""
        rate = np.random.uniform(*rate_range)
        indices = torch.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].long()
        stretched = audio[indices]
        
        # Pad or truncate to original length
        if len(stretched) < len(audio):
            padding = len(audio) - len(stretched)
            stretched = torch.cat([stretched, torch.zeros(padding)])
        else:
            stretched = stretched[:len(audio)]
            
        return stretched

class CryDataset(Dataset):
    def __init__(self, raw_audio, spectrograms, mfcc_features, spectral_features, 
                 rhythm_features, labels, transform=None):
        self.raw_audio = raw_audio
        self.spectrograms = spectrograms
        self.mfcc_features = mfcc_features
        self.spectral_features = spectral_features
        self.rhythm_features = rhythm_features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get data for this index
        try:
            raw_audio = torch.FloatTensor(self.raw_audio[idx])
            spectrogram = torch.FloatTensor(self.spectrograms[idx])
            mfcc = torch.FloatTensor(self.mfcc_features[idx])
            spectral = torch.FloatTensor(self.spectral_features[idx])
            rhythm = torch.FloatTensor(self.rhythm_features[idx])
            label = torch.LongTensor([self.labels[idx]])[0]
            
            # Validate tensor shapes
            if raw_audio.numel() == 0 or raw_audio.shape[0] == 0:
                print(f"Warning: Invalid raw_audio at index {idx}, shape: {raw_audio.shape}")
                # Return a valid tensor with zeros
                raw_audio = torch.zeros(80000)
            
            # Apply transforms if any
            if self.transform:
                raw_audio = self.transform(raw_audio)
                # Validate after transform
                if raw_audio.numel() == 0 or raw_audio.shape[0] == 0:
                    print(f"Warning: Transform made audio invalid at index {idx}")
                    raw_audio = torch.zeros(80000)
                
            return raw_audio, spectrogram, mfcc, spectral, rhythm, label
            
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return default values
            return (torch.zeros(80000), torch.zeros(128, 157), 
                   torch.zeros(156), torch.zeros(20), torch.zeros(4), 
                   torch.LongTensor([0])[0])

def create_data_loaders(features_dict, test_size=0.2, val_size=0.1, batch_size=32, 
                       num_workers=0, random_state=42):
    """Create train, validation, and test data loaders"""
    
    # Extract features
    raw_audio = features_dict['raw_audio']
    spectrograms = features_dict['spectrograms']
    mfcc_features = features_dict['mfcc']
    spectral_features = features_dict['spectral']
    rhythm_features = features_dict['rhythm']
    labels = features_dict['labels']
    
    # First split: train+val and test
    X_temp = list(zip(raw_audio, spectrograms, mfcc_features, spectral_features, rhythm_features))
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_temp, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_train_val
    )
    
    # Unpack the features
    def unpack_features(X):
        raw_audio, spectrograms, mfcc, spectral, rhythm = zip(*X)
        return np.array(raw_audio), np.array(spectrograms), np.array(mfcc), np.array(spectral), np.array(rhythm)
    
    # Train data
    train_raw_audio, train_spectrograms, train_mfcc, train_spectral, train_rhythm = unpack_features(X_train)
    
    # Validation data
    val_raw_audio, val_spectrograms, val_mfcc, val_spectral, val_rhythm = unpack_features(X_val)
    
    # Test data
    test_raw_audio, test_spectrograms, test_mfcc, test_spectral, test_rhythm = unpack_features(X_test)
    
    # Create datasets
    train_dataset = CryDataset(
        train_raw_audio, train_spectrograms, train_mfcc, train_spectral, train_rhythm, y_train
    )
    
    val_dataset = CryDataset(
        val_raw_audio, val_spectrograms, val_mfcc, val_spectral, val_rhythm, y_val
    )
    
    test_dataset = CryDataset(
        test_raw_audio, test_spectrograms, test_mfcc, test_spectral, test_rhythm, y_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)

def collate_fn(batch):
    """Custom collate function for handling variable length sequences"""
    raw_audio, spectrograms, mfcc, spectral, rhythm, labels = zip(*batch)
    
    # Filter out empty tensors and validate shapes
    valid_batch = []
    for i, (audio, spec, mfcc_feat, spec_feat, rhythm_feat, label) in enumerate(batch):
        # Check if audio has valid length
        if audio.numel() > 0 and audio.shape[0] > 0:
            valid_batch.append((audio, spec, mfcc_feat, spec_feat, rhythm_feat, label))
        else:
            print(f"Warning: Skipping batch item {i} with invalid audio shape: {audio.shape}")
    
    if len(valid_batch) == 0:
        # Return empty tensors if no valid items
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    # Unpack valid batch
    raw_audio, spectrograms, mfcc, spectral, rhythm, labels = zip(*valid_batch)
    
    # Stack tensors
    raw_audio = torch.stack(raw_audio)
    spectrograms = torch.stack(spectrograms)
    mfcc = torch.stack(mfcc)
    spectral = torch.stack(spectral)
    rhythm = torch.stack(rhythm)
    labels = torch.stack(labels)
    
    return raw_audio, spectrograms, mfcc, spectral, rhythm, labels

class DataAugmentation:
    """Advanced data augmentation for audio data"""
    
    def __init__(self, noise_prob=0.3, shift_prob=0.3, stretch_prob=0.3):
        self.noise_prob = noise_prob
        self.shift_prob = shift_prob
        self.stretch_prob = stretch_prob
    
    def __call__(self, audio):
        """Apply random augmentations"""
        if np.random.random() < self.noise_prob:
            audio = AudioTransforms.add_noise(audio)
        
        if np.random.random() < self.shift_prob:
            audio = AudioTransforms.time_shift(audio)
        
        if np.random.random() < self.stretch_prob:
            audio = AudioTransforms.time_stretch(audio)
        
        return audio

def create_augmented_loaders(features_dict, test_size=0.2, val_size=0.1, batch_size=32, 
                           num_workers=0, random_state=42):
    """Create data loaders with augmentation for training data"""
    
    train_loader, val_loader, test_loader, datasets = create_data_loaders(
        features_dict, test_size, val_size, batch_size, num_workers, random_state
    )
    
    # Add augmentation to training dataset
    train_dataset, val_dataset, test_dataset = datasets
    train_dataset.transform = DataAugmentation()
    
    # Recreate train loader with augmentation
    augmented_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    # Update other loaders with custom collate function
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    return augmented_train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    import joblib
    
    # Load processed data
    data = np.load('models/processed_data.npz')
    features_dict = {key: data[key] for key in data.files}
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_augmented_loaders(
        features_dict, batch_size=16, num_workers=0
    )
    
    # Test data loading
    for batch_idx, (raw_audio, spectrograms, mfcc, spectral, rhythm, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Raw audio shape: {raw_audio.shape}")
        print(f"  Spectrograms shape: {spectrograms.shape}")
        print(f"  MFCC shape: {mfcc.shape}")
        print(f"  Spectral shape: {spectral.shape}")
        print(f"  Rhythm shape: {rhythm.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        if batch_idx == 0:  # Just test first batch
            break