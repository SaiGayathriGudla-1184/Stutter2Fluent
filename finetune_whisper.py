import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import ctranslate2

# --- Configuration ---
BASE_MODEL = "openai/whisper-base"
OUTPUT_DIR = "whisper-finetuned-stutter"
FINAL_CT2_OUTPUT = os.path.join("models", "fine_tuned_whisper_ct2")

# --- Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def train():
    print(f"🚀 Starting Fine-Tuning for {BASE_MODEL}...")
    
    # 1. Load Processor & Model
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map="auto")
    
    # 2. Prepare Model for LoRA
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 3. Load Dataset
    # NOTE: Replace this with your actual dysfluency dataset (e.g., SEP-28K)
    # Here we use a dummy subset of Common Voice for demonstration
    print("📦 Loading dataset (using Common Voice tiny subset as placeholder)...")
    ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train[:100]", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    # Map dataset
    ds = ds.map(lambda x: prepare_dataset(x, processor), remove_columns=ds.column_names, num_proc=1)
    
    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=50,
        max_steps=200, # Increase for real training
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=100,
        logging_steps=25,
        report_to=["none"],
    )

    # 5. Train
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
    )
    
    trainer.train()
    
    # 6. Save Adapter
    print("💾 Saving LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    return model, processor

def convert_to_ct2():
    """
    Merges the LoRA adapter with the base model and converts it to CTranslate2 format.
    """
    print("\n🔄 Converting to CTranslate2 format for Vocal Agent...")
    
    # Reload base model in full precision for merging
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, device_map="cpu")
    from peft import PeftModel
    
    # Load adapters
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model = model.merge_and_unload()
    
    # Save merged model temporarily
    merged_dir = f"{OUTPUT_DIR}_merged"
    model.save_pretrained(merged_dir)
    
    # Load processor to save tokenizer
    processor = WhisperProcessor.from_pretrained(OUTPUT_DIR)
    processor.save_pretrained(merged_dir)

    # Convert to CTranslate2
    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=merged_dir,
        copy_files=["tokenizer.json", "preprocessor_config.json"]
    )
    
    os.makedirs(FINAL_CT2_OUTPUT, exist_ok=True)
    
    # Quantize to int8 for speed/size balance
    converter.convert(
        output_dir=FINAL_CT2_OUTPUT,
        quantization="int8",
        force=True
    )
    
    print(f"✅ Model converted and saved to: {FINAL_CT2_OUTPUT}")
    print("👉 Restart 'main.py' to use the new model.")

if __name__ == "__main__":
    # Check for GPU
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected. Training will be extremely slow.")
    
    try:
        train()
        convert_to_ct2()
    except Exception as e:
        print(f"❌ Error: {e}")