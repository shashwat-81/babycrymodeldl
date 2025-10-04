"""
Enhanced Test Data Generator for Baby Cry Classification

This script generates additional test data using various audio augmentation techniques
to expand the testing dataset for better model evaluation.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioAugmentation:
    """Audio augmentation class with various transformation techniques"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add white noise to audio"""
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio
    
    def time_stretch(self, audio, rate=1.0):
        """Change the speed of audio without changing pitch"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps=0):
        """Shift the pitch of audio"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def volume_change(self, audio, factor=1.0):
        """Change volume of audio"""
        return audio * factor
    
    def add_reverb(self, audio, room_size=0.5, damping=0.5):
        """Add simple reverb effect"""
        # Simple reverb simulation using delay and decay
        delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
        reverb = np.zeros(len(audio) + delay_samples)
        reverb[:len(audio)] += audio
        reverb[delay_samples:] += audio * room_size * damping
        return reverb[:len(audio)]
    
    def frequency_mask(self, audio, mask_fraction=0.1):
        """Apply frequency masking (spectral gating)"""
        # Convert to frequency domain
        stft = librosa.stft(audio)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Apply frequency mask
        freq_bins = magnitude.shape[0]
        mask_size = int(freq_bins * mask_fraction)
        start_bin = random.randint(0, freq_bins - mask_size)
        magnitude[start_bin:start_bin + mask_size, :] *= 0.1
        
        # Convert back to time domain
        stft_masked = magnitude * np.exp(1j * phase)
        return librosa.istft(stft_masked)
    
    def time_mask(self, audio, mask_fraction=0.1):
        """Apply time masking"""
        mask_length = int(len(audio) * mask_fraction)
        start_idx = random.randint(0, len(audio) - mask_length)
        masked_audio = audio.copy()
        masked_audio[start_idx:start_idx + mask_length] *= 0.1
        return masked_audio
    
    def harmonic_distortion(self, audio, factor=0.1):
        """Add harmonic distortion"""
        # Simple distortion by adding higher harmonics
        harmonic = np.sin(2 * np.pi * np.arange(len(audio)) / self.sample_rate * 2)
        return audio + factor * harmonic[:len(audio)]

class TestDataGenerator:
    """Generate enhanced test data from existing audio files"""
    
    def __init__(self, source_dir, output_dir, sample_rate=22050):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.augmenter = AudioAugmentation(sample_rate)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation strategies
        self.augmentation_strategies = {
            'noise_light': lambda x: self.augmenter.add_noise(x, 0.002),
            'noise_medium': lambda x: self.augmenter.add_noise(x, 0.005),
            'noise_heavy': lambda x: self.augmenter.add_noise(x, 0.01),
            'speed_fast': lambda x: self.augmenter.time_stretch(x, 1.2),
            'speed_slow': lambda x: self.augmenter.time_stretch(x, 0.8),
            'pitch_up': lambda x: self.augmenter.pitch_shift(x, 2),
            'pitch_down': lambda x: self.augmenter.pitch_shift(x, -2),
            'volume_up': lambda x: self.augmenter.volume_change(x, 1.3),
            'volume_down': lambda x: self.augmenter.volume_change(x, 0.7),
            'reverb_light': lambda x: self.augmenter.add_reverb(x, 0.3, 0.7),
            'reverb_heavy': lambda x: self.augmenter.add_reverb(x, 0.7, 0.5),
            'freq_mask': lambda x: self.augmenter.frequency_mask(x, 0.1),
            'time_mask': lambda x: self.augmenter.time_mask(x, 0.1),
            'harmonic_dist': lambda x: self.augmenter.harmonic_distortion(x, 0.05),
        }
    
    def load_audio(self, file_path):
        """Load audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def save_audio(self, audio, file_path):
        """Save audio file"""
        try:
            # Normalize audio to prevent clipping
            audio = np.clip(audio, -1.0, 1.0)
            sf.write(file_path, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False
    
    def generate_augmented_samples(self, class_name, num_samples_per_original=5):
        """Generate augmented samples for a specific class"""
        source_class_dir = self.source_dir / class_name
        output_class_dir = self.output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_class_dir.exists():
            print(f"Source directory {source_class_dir} does not exist!")
            return
        
        # Get all audio files
        audio_files = list(source_class_dir.glob('*.wav'))
        if not audio_files:
            print(f"No WAV files found in {source_class_dir}")
            return
        
        print(f"\\nGenerating augmented data for class: {class_name}")
        print(f"Found {len(audio_files)} original files")
        
        generated_count = 0
        
        for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
            # Load original audio
            audio = self.load_audio(audio_file)
            if audio is None:
                continue
            
            # Copy original file to test dataset
            original_name = f"{class_name}_original_{audio_file.stem}.wav"
            self.save_audio(audio, output_class_dir / original_name)
            generated_count += 1
            
            # Generate augmented versions
            strategies = random.sample(list(self.augmentation_strategies.keys()), 
                                     min(num_samples_per_original, len(self.augmentation_strategies)))
            
            for i, strategy_name in enumerate(strategies):
                try:
                    augmented_audio = self.augmentation_strategies[strategy_name](audio)
                    augmented_name = f"{class_name}_aug_{strategy_name}_{audio_file.stem}_{i+1}.wav"
                    
                    if self.save_audio(augmented_audio, output_class_dir / augmented_name):
                        generated_count += 1
                        
                except Exception as e:
                    print(f"Error applying {strategy_name} to {audio_file}: {e}")
        
        print(f"Generated {generated_count} samples for {class_name}")
        return generated_count
    
    def generate_synthetic_samples(self, class_name, num_samples=10):
        """Generate synthetic audio samples using signal processing"""
        output_class_dir = self.output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\\nGenerating synthetic samples for class: {class_name}")
        
        # Define class-specific frequency patterns (rough approximations)
        class_patterns = {
            'belly pain': {'base_freq': 400, 'variation': 100, 'duration_range': (1.0, 3.0)},
            'cold_hot': {'base_freq': 350, 'variation': 80, 'duration_range': (0.5, 2.0)},
            'discomfort': {'base_freq': 300, 'variation': 120, 'duration_range': (1.5, 4.0)},
            'hungry': {'base_freq': 450, 'variation': 150, 'duration_range': (2.0, 5.0)},
            'tired': {'base_freq': 250, 'variation': 60, 'duration_range': (1.0, 2.5)},
        }
        
        pattern = class_patterns.get(class_name, class_patterns['discomfort'])
        
        for i in tqdm(range(num_samples), desc=f"Generating synthetic {class_name}"):
            # Random duration
            duration = random.uniform(*pattern['duration_range'])
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            
            # Generate base cry pattern
            base_freq = pattern['base_freq'] + random.uniform(-pattern['variation'], pattern['variation'])
            
            # Create fundamental frequency with modulation
            modulation = np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
            frequency = base_freq * (1 + 0.1 * modulation)
            
            # Generate audio with harmonics
            audio = np.zeros_like(t)
            for harmonic in range(1, 6):  # First 5 harmonics
                amplitude = 1.0 / harmonic  # Decreasing amplitude
                audio += amplitude * np.sin(2 * np.pi * harmonic * frequency * t)
            
            # Add formant-like resonances
            for formant_freq in [800, 1200, 2500]:
                formant = np.sin(2 * np.pi * formant_freq * t) * np.exp(-t * 2)
                audio += 0.1 * formant
            
            # Apply envelope (attack, decay, sustain, release)
            envelope = np.ones_like(t)
            attack_time = int(0.1 * self.sample_rate)
            release_time = int(0.2 * self.sample_rate)
            
            if len(envelope) > attack_time:
                envelope[:attack_time] = np.linspace(0, 1, attack_time)
            if len(envelope) > release_time:
                envelope[-release_time:] = np.linspace(1, 0, release_time)
            
            audio *= envelope
            
            # Add noise and variations
            noise = np.random.normal(0, 0.05, len(audio))
            audio += noise
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save synthetic sample
            synthetic_name = f"{class_name}_synthetic_{i+1:03d}.wav"
            self.save_audio(audio, output_class_dir / synthetic_name)
        
        print(f"Generated {num_samples} synthetic samples for {class_name}")
        return num_samples
    
    def generate_mixed_samples(self, num_samples=5):
        """Generate mixed/transition samples between classes"""
        classes = ['belly pain', 'cold_hot', 'discomfort', 'hungry', 'tired']
        mixed_dir = self.output_dir / 'mixed_samples'
        mixed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\\nGenerating mixed/transition samples")
        
        generated_count = 0
        
        for i in tqdm(range(num_samples), desc="Generating mixed samples"):
            # Select two random classes
            class1, class2 = random.sample(classes, 2)
            
            # Try to find audio files from both classes
            class1_files = list((self.source_dir / class1).glob('*.wav'))
            class2_files = list((self.source_dir / class2).glob('*.wav'))
            
            if class1_files and class2_files:
                # Load random files from each class
                audio1 = self.load_audio(random.choice(class1_files))
                audio2 = self.load_audio(random.choice(class2_files))
                
                if audio1 is not None and audio2 is not None:
                    # Make them the same length
                    min_length = min(len(audio1), len(audio2))
                    audio1 = audio1[:min_length]
                    audio2 = audio2[:min_length]
                    
                    # Create transition: start with class1, fade to class2
                    fade_length = min_length // 4
                    mixed_audio = audio1.copy()
                    
                    # Create crossfade
                    fade_in = np.linspace(0, 1, fade_length)
                    fade_out = np.linspace(1, 0, fade_length)
                    
                    start_fade = min_length // 2
                    end_fade = start_fade + fade_length
                    
                    if end_fade <= min_length:
                        mixed_audio[start_fade:end_fade] = (
                            audio1[start_fade:end_fade] * fade_out + 
                            audio2[start_fade:end_fade] * fade_in
                        )
                        mixed_audio[end_fade:] = audio2[end_fade:]
                    
                    # Save mixed sample
                    mixed_name = f"mixed_{class1.replace(' ', '_')}_{class2.replace(' ', '_')}_{i+1}.wav"
                    if self.save_audio(mixed_audio, mixed_dir / mixed_name):
                        generated_count += 1
        
        print(f"Generated {generated_count} mixed samples")
        return generated_count
    
    def generate_all_test_data(self, samples_per_original=3, synthetic_per_class=5, mixed_samples=10):
        """Generate complete test dataset"""
        print("=== Baby Cry Test Data Generator ===")
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        
        total_generated = 0
        
        # Define classes
        classes = ['belly pain', 'cold_hot', 'discomfort', 'hungry', 'tired']
        
        # Generate augmented samples for each class
        for class_name in classes:
            count = self.generate_augmented_samples(class_name, samples_per_original)
            total_generated += count
        
        # Generate synthetic samples
        for class_name in classes:
            count = self.generate_synthetic_samples(class_name, synthetic_per_class)
            total_generated += count
        
        # Generate mixed samples
        count = self.generate_mixed_samples(mixed_samples)
        total_generated += count
        
        print(f"\\n=== Generation Complete ===")
        print(f"Total samples generated: {total_generated}")
        
        # Generate summary report
        self.generate_summary_report()
        
        return total_generated
    
    def generate_summary_report(self):
        """Generate a summary report of the test dataset"""
        report_path = self.output_dir / "dataset_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Baby Cry Test Dataset Summary\\n")
            f.write("=" * 40 + "\\n\\n")
            
            total_files = 0
            for class_dir in self.output_dir.iterdir():
                if class_dir.is_dir():
                    audio_files = list(class_dir.glob('*.wav'))
                    f.write(f"{class_dir.name}: {len(audio_files)} files\\n")
                    total_files += len(audio_files)
            
            f.write(f"\\nTotal audio files: {total_files}\\n")
            f.write(f"Generated on: {os.popen('date').read().strip()}\\n")
        
        print(f"Dataset summary saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate test data for baby cry classification')
    parser.add_argument('--source', '-s', type=str, default='Data', 
                       help='Source directory containing original audio files')
    parser.add_argument('--output', '-o', type=str, default='enhanced_test_dataset', 
                       help='Output directory for generated test data')
    parser.add_argument('--samples-per-original', type=int, default=3,
                       help='Number of augmented samples per original file')
    parser.add_argument('--synthetic-per-class', type=int, default=5,
                       help='Number of synthetic samples per class')
    parser.add_argument('--mixed-samples', type=int, default=10,
                       help='Number of mixed/transition samples to generate')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Audio sample rate')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TestDataGenerator(
        source_dir=args.source,
        output_dir=args.output,
        sample_rate=args.sample_rate
    )
    
    # Generate all test data
    total_generated = generator.generate_all_test_data(
        samples_per_original=args.samples_per_original,
        synthetic_per_class=args.synthetic_per_class,
        mixed_samples=args.mixed_samples
    )
    
    print(f"\\nTest data generation completed! Generated {total_generated} total samples.")
    print(f"Check the '{args.output}' directory for the generated test data.")

if __name__ == "__main__":
    main()