import os
import torch
from faster_whisper import WhisperModel

# --- Configuration ---
# Path to your test data folder
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
MODEL_SIZE = "large-v3"

def generate_draft_transcripts():
    """
    Scans the test_data directory for .wav files.
    If a matching .txt file does not exist, it creates one
    and pre-fills it with a draft transcription from Whisper.
    """
    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ Directory not found: {TEST_DATA_DIR}")
        print("   Please create a folder named 'test_data' and put your .wav files in it.")
        return

    # Load model for pre-filling (saves typing time)
    print("⏳ Loading Whisper model to generate draft transcripts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = WhisperModel(MODEL_SIZE, device=device, compute_type="int8")
    except Exception as e:
        print(f"⚠️ Could not load model ({e}). Creating empty text files instead.")
        model = None

    wav_files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(".wav")]
    
    if not wav_files:
        print("❌ No .wav files found in test_data folder.")
        return

    print(f"🔎 Found {len(wav_files)} audio files.")

    for wav_file in wav_files:
        # Determine text filename (e.g., recording.wav -> recording.txt)
        txt_file = os.path.splitext(wav_file)[0] + ".txt"
        wav_path = os.path.join(TEST_DATA_DIR, wav_file)
        txt_path = os.path.join(TEST_DATA_DIR, txt_file)

        if os.path.exists(txt_path):
            print(f"✅ Exists: {txt_file} (Skipping)")
            continue

        print(f"📝 Creating draft for: {wav_file} ...")
        transcript_text = ""
        if model:
            try:
                segments, _ = model.transcribe(wav_path)
                transcript_text = "".join([s.text for s in segments]).strip()
            except Exception as e:
                print(f"   ⚠️ Transcription failed: {e}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        print(f"   ✨ Created {txt_file}")

    print("\n✅ Done! Steps to finish:")
    print("1. Go to the 'test_data' folder.")
    print("2. Open each .txt file.")
    print("3. CORRECT the text to match exactly what is said in the audio (Ground Truth).")
    print("4. Save the files.")
    print("5. Run 'python evaluate_stt.py' to test accuracy.")

if __name__ == "__main__":
    generate_draft_transcripts()