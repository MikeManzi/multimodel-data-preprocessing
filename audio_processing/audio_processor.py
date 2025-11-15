import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm


class AudioProcessor:
    def __init__(self, audio_dir: str, output_csv: str, sample_rate: int = 22050):
        self.audio_dir = Path(audio_dir)
        self.output_csv = Path(output_csv)
        self.sample_rate = sample_rate
        self.audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']

    def get_audio_files(self) -> List[Path]:
        audio_files = []
        for ext in self.audio_extensions:
            audio_files.extend(list(self.audio_dir.glob(ext)))
        return sorted(audio_files)

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def apply_pitch_shift(self, y: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    def apply_time_stretch(self, y: np.ndarray, rate: float = 1.2) -> np.ndarray:
        return librosa.effects.time_stretch(y, rate=rate)

    def add_background_noise(self, y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        noise = np.random.randn(len(y))
        return y + noise_factor * noise

    def apply_augmentations(self, y: np.ndarray, sr: int) -> List[Tuple[np.ndarray, str]]:
        augmentations = []

        augmentations.append((y.copy(), 'original'))

        pitch_shifted = self.apply_pitch_shift(y, sr, n_steps=2.0)
        augmentations.append((pitch_shifted, 'pitch_shift'))

        time_stretched = self.apply_time_stretch(y, rate=1.2)
        augmentations.append((time_stretched, 'time_stretch'))

        noisy = self.add_background_noise(y, noise_factor=0.005)
        augmentations.append((noisy, 'background_noise'))

        return augmentations

    def extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> float:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        return np.mean(rolloff)

    def extract_energy(self, y: np.ndarray) -> float:
        rms = librosa.feature.rms(y=y)
        return np.mean(rms)

    def extract_zero_crossing_rate(self, y: np.ndarray) -> float:
        zcr = librosa.feature.zero_crossing_rate(y)
        return np.mean(zcr)

    def extract_spectral_centroid(self, y: np.ndarray, sr: int) -> float:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(centroid)

    def extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        mfcc_features = self.extract_mfcc(y, sr)
        spectral_rolloff = self.extract_spectral_rolloff(y, sr)
        energy = self.extract_energy(y)
        zcr = self.extract_zero_crossing_rate(y)
        spectral_centroid = self.extract_spectral_centroid(y, sr)

        return np.concatenate([
            mfcc_features,
            [spectral_rolloff, energy, zcr, spectral_centroid]
        ])

    def process_all_audio(self) -> pd.DataFrame:
        audio_files = self.get_audio_files()
        print(f"Found {len(audio_files)} audio files to process")

        if len(audio_files) == 0:
            print("No audio files found. Creating empty dataframe structure.")
            return self._create_empty_dataframe()

        all_data = []
        expected_feature_count = None

        for audio_path in tqdm(audio_files, desc="Processing audio"):
            try:
                y, sr = self.load_audio(audio_path)
                filename = audio_path.name

                augmentations = self.apply_augmentations(y, sr)

                for aug_audio, aug_type in augmentations:
                    features = self.extract_features(aug_audio, sr)

                    if expected_feature_count is None:
                        expected_feature_count = len(features)
                        print(f"Feature vector length: {expected_feature_count}")

                    row_data = {
                        'filename': filename,
                        'augmentation': aug_type,
                        'duration': len(aug_audio) / sr,
                        'sample_rate': sr
                    }

                    for i, feat_val in enumerate(features):
                        row_data[f'feature_{i}'] = feat_val

                    all_data.append(row_data)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

        df = pd.DataFrame(all_data)

        if len(df) > 0:
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            nan_count = df[feature_cols].isna().sum().sum()
            print(f"NaN values in features: {nan_count}")

        df.to_csv(self.output_csv, index=False)

        print(f"Processing complete!")
        print(f"Total rows: {len(df)}")
        print(f"CSV saved to: {self.output_csv}")

        return df

    def _create_empty_dataframe(self) -> pd.DataFrame:
        columns = ['filename', 'augmentation', 'duration', 'sample_rate']
        for i in range(30):
            columns.append(f'feature_{i}')
        return pd.DataFrame(columns=columns)
