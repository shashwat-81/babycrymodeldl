import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        # attended_output shape: (batch_size, hidden_dim)
        
        return attended_output, attention_weights

class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super(Wav2VecFeatureExtractor, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Freeze wav2vec2 parameters initially
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
    def forward(self, audio_input):
        # audio_input shape: (batch_size, audio_length)
        with torch.no_grad():
            wav2vec_output = self.wav2vec2(audio_input).last_hidden_state
        return wav2vec_output

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25)
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        features = self.conv_layers(x)
        # Flatten for further processing
        features = features.view(features.size(0), -1)
        return features

class HybridCryClassifier(nn.Module):
    def __init__(self, num_classes=5, mfcc_dim=156, spectral_dim=20, rhythm_dim=4, 
                 spectrogram_shape=(128, 313), hidden_dim=128, lstm_layers=1):
        super(HybridCryClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_checkpointing = True
        
        # Simpler, more focused architecture
        # CNN for spectrogram processing (reduced complexity)
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3)
        )
        
        cnn_output_dim = 128 * 4 * 4
        
        # Traditional feature processing with better normalization
        self.mfcc_processor = nn.Sequential(
            nn.LayerNorm(mfcc_dim),
            nn.Linear(mfcc_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        self.spectral_processor = nn.Sequential(
            nn.LayerNorm(spectral_dim),
            nn.Linear(spectral_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8)
        )
        
        self.rhythm_processor = nn.Sequential(
            nn.LayerNorm(rhythm_dim),
            nn.Linear(rhythm_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        # Simple audio statistics instead of complex Wav2Vec
        self.audio_processor = nn.Sequential(
            nn.Linear(8, 32),  # 8 statistical features from raw audio
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Feature fusion with better architecture
        fusion_dim = cnn_output_dim + 32 + 8 + 4 + 16  # All features combined
        
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, raw_audio, spectrograms, mfcc_features, spectral_features, rhythm_features):
        batch_size = raw_audio.size(0)
        
        # 1. Extract statistical features from raw audio (much simpler than Wav2Vec)
        audio_stats = torch.stack([
            torch.mean(raw_audio, dim=1),
            torch.std(raw_audio, dim=1),
            torch.max(raw_audio, dim=1)[0],
            torch.min(raw_audio, dim=1)[0],
            torch.median(raw_audio, dim=1)[0],
            torch.var(raw_audio, dim=1),
            torch.norm(raw_audio, dim=1),
            torch.sum(torch.abs(raw_audio), dim=1) / raw_audio.shape[1]  # Mean absolute value
        ], dim=1)  # Shape: (batch_size, 8)
        
        audio_processed = self.audio_processor(audio_stats)
        
        # 2. CNN processing for spectrograms
        if len(spectrograms.shape) == 3:
            spectrograms = spectrograms.unsqueeze(1)
        cnn_features = self.cnn_extractor(spectrograms)
        cnn_features = cnn_features.view(batch_size, -1)
        
        # 3. Traditional feature processing
        mfcc_processed = self.mfcc_processor(mfcc_features)
        spectral_processed = self.spectral_processor(spectral_features)
        rhythm_processed = self.rhythm_processor(rhythm_features)
        
        # 4. Feature fusion
        fused_features = torch.cat([
            audio_processed,
            cnn_features,
            mfcc_processed,
            spectral_processed,
            rhythm_processed
        ], dim=1)
        
        # 5. Final processing
        processed_features = self.fusion_layer(fused_features)
        output = self.classifier(processed_features)
        
        return output, None  # No attention weights in this simplified version

class EnsembleClassifier(nn.Module):
    def __init__(self, num_classes=5, num_models=3):
        super(EnsembleClassifier, self).__init__()
        self.models = nn.ModuleList([
            HybridCryClassifier(num_classes) for _ in range(num_models)
        ])
        
    def forward(self, raw_audio, spectrograms, mfcc_features, spectral_features, rhythm_features):
        outputs = []
        attention_weights_list = []
        
        for model in self.models:
            output, attention_weights = model(raw_audio, spectrograms, mfcc_features, 
                                            spectral_features, rhythm_features)
            outputs.append(output)
            attention_weights_list.append(attention_weights)
        
        # Ensemble prediction (average)
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output, attention_weights_list

class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001, class_weights=None):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Use class weights if provided for handling imbalanced data
        if class_weights is not None:
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)
            # Use label smoothing for better generalization
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
            print(f"Using weighted loss with class weights and label smoothing: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )
        
        # Early stopping parameters
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 15
        
        # Mixed precision training for memory efficiency
        self.use_mixed_precision = device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training for memory optimization")
        else:
            self.scaler = None
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (raw_audio, spectrograms, mfcc, spectral, rhythm, labels) in enumerate(train_loader):
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Move data to device
            raw_audio = raw_audio.to(self.device, non_blocking=True)
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            mfcc = mfcc.to(self.device, non_blocking=True)
            spectral = spectral.to(self.device, non_blocking=True)
            rhythm = rhythm.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs, _ = self.model(raw_audio, spectrograms, mfcc, spectral, rhythm)
                    loss = self.criterion(outputs, labels)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                outputs, _ = self.model(raw_audio, spectrograms, mfcc, spectral, rhythm)
                loss = self.criterion(outputs, labels)
                
                # Regular backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for raw_audio, spectrograms, mfcc, spectral, rhythm, labels in val_loader:
                # Move data to device
                raw_audio = raw_audio.to(self.device, non_blocking=True)
                spectrograms = spectrograms.to(self.device, non_blocking=True)
                mfcc = mfcc.to(self.device, non_blocking=True)
                spectral = spectral.to(self.device, non_blocking=True)
                rhythm = rhythm.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed precision inference
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(raw_audio, spectrograms, mfcc, spectral, rhythm)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs, _ = self.model(raw_audio, spectrograms, mfcc, spectral, rhythm)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs, save_path='models/best_model.pth'):
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            print(f'\\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, save_path)
                print(f'Model saved with validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Plot training history
        self.plot_training_history()
        
        return val_preds, val_labels
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('web_app/static/training_history.png')
        plt.show()
    
    def evaluate_model(self, predictions, true_labels, label_encoder):
        """Evaluate model performance"""
        # Classification report
        class_names = label_encoder.classes_
        report = classification_report(true_labels, predictions, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('web_app/static/confusion_matrix.png')
        plt.show()
        
        return report, cm