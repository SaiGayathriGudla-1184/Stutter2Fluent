import os
import jiwer
import torch
from faster_whisper import WhisperModel

# --- Configuration ---
# Ensure this matches the model size used in main.py
MODEL_SIZE = "base" 
# Create a folder named 'test_data' and put pairs of .wav and .txt files in it
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

def calculate_wer():
    """
    Calculates Word Error Rate (WER) by comparing Whisper's transcription
    against manually verified text files.
    """
    # 1. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Whisper model '{MODEL_SIZE}' on {device} for evaluation...")
    
    try:
        model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Check Data Directory
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
        print(f"⚠️ Created directory: {TEST_DATA_DIR}")
        print("   Please add .wav audio files and matching .txt reference files there to test.")
        return

    ground_truths = []
    hypotheses = []

    print(f"\n{'Filename':<30} | {'WER':<10} | {'Status'}")
    print("-" * 60)

    # 3. Iterate through test files
    files_found = False
    for filename in os.listdir(TEST_DATA_DIR):
        if filename.endswith(".wav"):
            files_found = True
            audio_path = os.path.join(TEST_DATA_DIR, filename)
            text_path = os.path.join(TEST_DATA_DIR, filename.replace(".wav", ".txt"))

            if os.path.exists(text_path):
                # Read Ground Truth (Reference)
                with open(text_path, "r", encoding="utf-8") as f:
                    reference = f.read().strip()
                
                # Run Transcription (Hypothesis)
                segments, _ = model.transcribe(audio_path)
                hypothesis = "".join([s.text for s in segments]).strip()

                # Store results
                ground_truths.append(reference)
                hypotheses.append(hypothesis)

                # Calculate WER for this specific file
                error = jiwer.wer(reference, hypothesis)
                print(f"{filename:<30} | {error:.2%}    | ✅ Processed")
            else:
                print(f"{filename:<30} | N/A        | ⚠️ Missing .txt file")

    # 4. Summary
    if ground_truths:
        total_wer = jiwer.wer(ground_truths, hypotheses)
        print("-" * 60)
        print(f"📊 Overall Average WER: {total_wer:.2%}")
        print("(Lower is better. 0% means perfect transcription.)")
    elif not files_found:
        print("❌ No .wav files found in 'test_data' directory.")

if __name__ == "__main__":
    calculate_wer()
import os
import jiwer
import torch
from faster_whisper import WhisperModel

# --- Configuration ---
# Ensure this matches the model size used in main.py
MODEL_SIZE = "large-v3" 
# Create a folder named 'test_data' and put pairs of .wav and .txt files in it
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

def calculate_wer():
    """
    Calculates Word Error Rate (WER) by comparing Whisper's transcription
    against manually verified text files.
    """
    # 1. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Whisper model '{MODEL_SIZE}' on {device} for evaluation...")
    
    try:
        model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Check Data Directory
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
        print(f"⚠️ Created directory: {TEST_DATA_DIR}")
        print("   Please add .wav audio files and matching .txt reference files there to test.")
        return

    ground_truths = []
    hypotheses = []

    print(f"\n{'Filename':<30} | {'WER':<10} | {'Status'}")
    print("-" * 60)

    # 3. Iterate through test files
    files_found = False
    for filename in os.listdir(TEST_DATA_DIR):
        if filename.endswith(".wav"):
            files_found = True
            audio_path = os.path.join(TEST_DATA_DIR, filename)
            text_path = os.path.join(TEST_DATA_DIR, filename.replace(".wav", ".txt"))

            if os.path.exists(text_path):
                # Read Ground Truth (Reference)
                with open(text_path, "r", encoding="utf-8") as f:
                    reference = f.read().strip()
                
                # Run Transcription (Hypothesis)
                segments, _ = model.transcribe(audio_path)
                hypothesis = "".join([s.text for s in segments]).strip()

                # Store results
                ground_truths.append(reference)
                hypotheses.append(hypothesis)

                # Calculate WER for this specific file
                error = jiwer.wer(reference, hypothesis)
                print(f"{filename:<30} | {error:.2%}    | ✅ Processed")
            else:
                print(f"{filename:<30} | N/A        | ⚠️ Missing .txt file")

    # 4. Summary
    if ground_truths:
        total_wer = jiwer.wer(ground_truths, hypotheses)
        print("-" * 60)
        print(f"📊 Overall Average WER: {total_wer:.2%}")
        print("(Lower is better. 0% means perfect transcription.)")
    elif not files_found:
        print("❌ No .wav files found in 'test_data' directory.")

if __name__ == "__main__":
    calculate_wer()
