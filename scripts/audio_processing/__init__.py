from .audio_processor import AudioProcessor
from .visualizer import AudioVisualizer
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple

# Convenience functions for CLI app
def load_audio(file_path: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate."""
    y, sr = librosa.load(file_path, sr=sample_rate)
    return y, sr

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract audio features from audio data."""
    # Create a temporary AudioProcessor instance to use its feature extraction
    processor = AudioProcessor(".", "temp.csv", sr)
    return processor.extract_features(y, sr)

__all__ = ['AudioProcessor', 'AudioVisualizer', 'load_audio', 'extract_features']
