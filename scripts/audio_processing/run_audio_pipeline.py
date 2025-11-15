from pathlib import Path
from audio_processor import AudioProcessor
from visualizer import AudioVisualizer


def main():
    ROOT = Path(__file__).parent.parent
    AUDIO_DIR = ROOT / 'audio_data'
    OUTPUT_DIR = ROOT / 'audio_visualizations'
    OUTPUT_CSV = ROOT / 'audio_features.csv'

    AUDIO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("extract audio features")
    processor = AudioProcessor(
        audio_dir=str(AUDIO_DIR),
        output_csv=str(OUTPUT_CSV),
        sample_rate=22050
    )

    df = processor.process_all_audio()

    if len(df) > 0:
        print("\nDataframe info:")
        print(f"Shape: {df.shape}")
        print(f"\nAugmentation counts:")
        print(df['augmentation'].value_counts())
        print(f"\nFirst few rows:")
        print(df.head())

    print("audio visualization")
    visualizer = AudioVisualizer(
        audio_dir=str(AUDIO_DIR),
        output_dir=str(OUTPUT_DIR),
        sample_rate=22050
    )

    visualizer.visualize_all_audio()

    print(f"Features CSV: {OUTPUT_CSV}")
    print(f"Visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
