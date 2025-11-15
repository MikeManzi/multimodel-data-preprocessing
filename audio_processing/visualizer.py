import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


class AudioVisualizer:
    def __init__(self, audio_dir: str, output_dir: str, sample_rate: int = 22050):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate

    def get_audio_files(self) -> List[Path]:
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(self.audio_dir.glob(ext)))
        return sorted(audio_files)

    def plot_waveform(self, y: np.ndarray, sr: int, title: str, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        return ax

    def plot_spectrogram(self, y: np.ndarray, sr: int, title: str, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(title)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')

        return ax

    def visualize_audio_file(self, file_path: Path):
        y, sr = librosa.load(file_path, sr=self.sample_rate)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Audio Analysis: {file_path.name}', fontsize=16)

        self.plot_waveform(y, sr, 'Waveform', ax=axes[0])
        self.plot_spectrogram(y, sr, 'Spectrogram', ax=axes[1])

        plt.tight_layout()

        output_path = self.output_dir / f'{file_path.stem}_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {output_path}")

        return output_path

    def visualize_all_audio(self):
        audio_files = self.get_audio_files()

        if len(audio_files) == 0:
            print("No audio files found for visualization.")
            return

        print(f"Found {len(audio_files)} audio files to visualize")

        for audio_path in audio_files:
            try:
                self.visualize_audio_file(audio_path)
            except Exception as e:
                print(f"Error visualizing {audio_path}: {e}")
                continue

        print(f"Visualization complete! Saved to {self.output_dir}")

    def plot_combined_waveforms(self, file_paths: List[Path], member_names: List[str] = None):
        n_files = len(file_paths)

        if n_files == 0:
            print("No audio files provided for combined visualization.")
            return

        fig, axes = plt.subplots(n_files, 1, figsize=(14, 3 * n_files))

        if n_files == 1:
            axes = [axes]

        for idx, (file_path, ax) in enumerate(zip(file_paths, axes)):
            try:
                y, sr = librosa.load(file_path, sr=self.sample_rate)

                if member_names and idx < len(member_names):
                    title = f'{member_names[idx]} - {file_path.name}'
                else:
                    title = file_path.name

                self.plot_waveform(y, sr, title, ax=ax)

            except Exception as e:
                print(f"Error plotting {file_path}: {e}")
                continue

        plt.tight_layout()

        output_path = self.output_dir / 'combined_waveforms.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved combined waveforms: {output_path}")

        return output_path
