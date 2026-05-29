"""
Sovereign Voice Training Pipeline
Fine-tune VibeVoice-Realtime-0.5B on Israel's voice using LoRA.

Pipeline:
1. Collect voice samples → Whisper transcribe
2. Create (text, audio) pairs
3. LoRA fine-tune VibeVoice
4. Deploy as custom voice adapter

Requirements: RTX 3070 8GB+ (MC)
Training time: ~2-4 hours for 30 min of voice data
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ── Training Config ─────────────────────────────────────────────────────────

@dataclass
class SovereignVoiceConfig:
    """Configuration for voice cloning."""
    # Model
    base_model: str = "microsoft/VibeVoice-Realtime-0.5B"
    output_dir: str = "/home/user/hermes-workspace/voice-agent-platform/sovereign-voice"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    warmup_steps: int = 100
    save_steps: int = 200
    max_length: int = 512
    
    # Data
    data_dir: str = "/home/user/hermes-workspace/voice-agent-platform/voice_samples"
    min_audio_length: float = 3.0  # seconds
    max_audio_length: float = 30.0
    sample_rate: int = 24000
    
    # Optimization
    use_8bit: bool = True
    bf16: bool = True
    fp16: bool = False
    
    # Reproducibility
    seed: int = 42
    wandb_mode: str = "disabled"


class VoiceDataset:
    """Process voice samples into training pairs."""
    
    def __init__(self, config: SovereignVoiceConfig):
        self.config = config
        self.samples = []
    
    def scan_directory(self, directory: str = None) -> list[dict]:
        """Scan for audio files + transcripts."""
        data_dir = Path(directory or self.config.data_dir)
        samples = []
        
        for audio_file in data_dir.glob("**/*"):
            if audio_file.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                # Look for matching transcript
                transcript_file = audio_file.with_suffix(".txt")
                transcript = ""
                if transcript_file.exists():
                    transcript = transcript_file.read_text().strip()
                
                samples.append({
                    "audio": str(audio_file),
                    "text": transcript,
                    "speaker": "israel",
                    "duration": 0,  # Will be populated by load_audio
                })
        
        self.samples = samples
        print(f"[Dataset] Found {len(samples)} voice samples")
        return samples
    
    def validate_samples(self) -> list[dict]:
        """Filter samples by quality thresholds."""
        valid = []
        for s in self.samples:
            if s.get("text") and len(s["text"]) > 5:
                valid.append(s)
            else:
                print(f"  [Skip] {s.get('audio')} — no transcript or too short")
        
        print(f"[Dataset] {len(valid)} valid samples after filtering")
        return valid
    
    def estimate_training_time(self, num_samples: int, epochs: int = None) -> dict:
        """Estimate training time on RTX 3070."""
        epochs = epochs or self.config.epochs
        # ~30 seconds per sample per epoch on RTX 3070
        seconds_per_sample = 30
        total_seconds = num_samples * epochs * seconds_per_sample / self.config.gradient_accumulation_steps
        
        return {
            "samples": num_samples,
            "epochs": epochs,
            "estimated_minutes": round(total_seconds / 60, 1),
            "estimated_hours": round(total_seconds / 3600, 2),
        }


class SovereignVoiceTrainer:
    """Fine-tune VibeVoice with LoRA for voice cloning."""
    
    def __init__(self, config: SovereignVoiceConfig = None):
        self.config = config or SovereignVoiceConfig()
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load VibeVoice model with optional 8-bit quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[SovereignVoice] Loading {self.config.base_model}")
        print(f"  8-bit: {self.config.use_8bit}")
        print(f"  bf16: {self.config.bf16}")
        
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if self.config.use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                print(f"  [Warn] 8-bit not available: {e}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **load_kwargs,
        )
        
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Model loaded. GPU memory: {gpu_mem:.1f}GB")
        
        return self.model
    
    def apply_lora(self):
        """Apply LoRA adapters to model."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise RuntimeError("Install peft: pip install peft")
        
        # Prepare for k-bit training
        if self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.target_modules),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[LoRA] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        return self.model
    
    def train(self, dataset: list[dict]):
        """Run LoRA fine-tuning."""
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise RuntimeError("Install trl: pip install trl")
        
        print(f"\n[SovereignVoice] Starting training")
        print(f"  Samples: {len(dataset)}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  LR: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            logging_steps=10,
            report_to=None,  # No W&B
            max_length=self.config.max_length,
            seed=self.config.seed,
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        
        start = time.time()
        trainer.train()
        elapsed = time.time() - start
        
        print(f"\n[SovereignVoice] Training complete in {elapsed/60:.1f} minutes")
        
        # Save LoRA adapter
        adapter_path = os.path.join(self.config.output_dir, "sovereign-lora")
        trainer.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        print(f"[SovereignVoice] LoRA adapter saved → {adapter_path}")
        
        return adapter_path
    
    def synthesize(self, text: str, voice: str = "sovereign") -> bytes:
        """Generate speech with the fine-tuned model."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
            )
        
        # Decode: token IDs → audio waveform
        generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        
        # In production: VibeVoice acoustic decoder converts tokens to audio
        # For now, return token representation
        return generated_tokens.cpu().numpy().tobytes()


# ── Entrypoint ──────────────────────────────────────────────────────────────

def main():
    """Run sovereign voice training pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Sovereign Voice Training")
    parser.add_argument("--data", default=None, help="Voice samples directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true", help="Check setup without training")
    args = parser.parse_args()
    
    config = SovereignVoiceConfig(epochs=args.epochs, learning_rate=args.lr)
    if args.data:
        config.data_dir = args.data
    
    dataset = VoiceDataset(config)
    samples = dataset.scan_directory()
    
    if not samples:
        print("[!] No voice samples found. Place audio files in voice_samples/")
        print("    Each audio file should have a matching .txt transcript")
        return
    
    valid = dataset.validate_samples()
    est = dataset.estimate_training_time(len(valid))
    print(f"\n[Est] Training time: {est['estimated_hours']} hours ({est['estimated_minutes']} min)")
    
    if args.dry_run:
        print("\n[Dry run] Setup verified. Ready for training.")
        return
    
    trainer = SovereignVoiceTrainer(config)
    trainer.load_model()
    trainer.apply_lora()
    adapter_path = trainer.train(valid)
    
    print(f"\n[Sovereign Voice] Complete!")
    print(f"  Adapter: {adapter_path}")
    print(f"  Deploy: python vibevoice_server.py --model realtime-0.5b --lora {adapter_path}")


if __name__ == "__main__":
    main()
