import os
import time
from PIL import Image
import librosa
from image_processing import process_image
from audio_processing import extract_features as extract_audio, load_audio
from prediction_helpers import predict_face, predict_voice, predict_product

def loader(message="Processing", dots=7, delay=0.5):
    print(message, end="", flush=True)
    for _ in range(dots):
        time.sleep(delay)
        print(".", end="", flush=True)
    print()

def run():
    print("\n===== WELCOME TO THE CLI MINI APP =====\n")

    # === STEP 1: Facial Authentication ===
    while True:
        print("\n----- STEP 1: Facial Recognition -----")
        img_path = input("Enter path to image: ").strip()
        if img_path == "":
            print("ACCESS DENIED: No image provided.")
            continue

        if not os.path.exists(img_path):
            print("Invalid path: File not found.")
            continue

        try:
            loader("Scanning face")
            img_feats = process_image(img_path)
        except Exception as e:
            print("Invalid image: Please provide a valid image file.")
            print("Error:", e)
            continue

        face_conf = predict_face(img_feats)

        if face_conf < 20:
            print(f"ACCESS DENIED: Face not recognized ({face_conf:.2f}%)")
            continue

        print(f"Facial Recognition Passed ({face_conf:.2f}%)")
        break

    # === STEP 2: Product Recommendation ===
    print("\n----- STEP 2: Product Recommendation -----")
    loader("Generating recommendation")
    product = predict_product()
    print(f"Product recommendation ready")

    # === STEP 3: Voice Verification ===
    print("\n----- STEP 3: Voice Verification -----")
    while True:
        audio_path = input("Enter path to audio: ").strip()
        if not os.path.exists(audio_path):
            print("Audio file not found. Try again.")
            continue

        loader("Verifying voice")
        try:
            y, sr = load_audio(audio_path)
            audio_feats = extract_audio(y, sr)
            voice_conf = predict_voice(audio_feats)
        except Exception as e:
            print(f"[!] Failed to load audio: {type(e).__name__} - {str(e) or 'No details provided'}")
            continue

        if voice_conf < 50:
            print(f"ACCESS DENIED: Voice not verified ({voice_conf:.2f}%). Try again.")
            continue

        print(f"Voice Verified ({voice_conf:.2f}%)")
        break

    print("\n===== ACCESS GRANTED =====")
    print(f"Recommended Product: {product}")

if __name__ == "__main__":
    run()
