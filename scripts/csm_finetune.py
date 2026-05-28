#!/usr/bin/env python3
"""
CSM-1B Voice Fine-Tuning Pipeline for MC (Windows + RTX 3070)
Run this on MC after voice data has been extracted.

Prerequisites:
1. Extract voice data using youtube_voice_extract.py on sandbox
2. Copy voice_data/ folder to MC
3. Install: pip install torch torchaudio transformers moshi-ml accelerate
4. Download CSM-1B: huggingface-cli download sesame/csm-1b

Usage:
    python csm_finetune.py --data-dir ./voice_data/israel --output-dir ./csm-israel --epochs 25
"""

import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("csm-finetune")

class VoiceDataset(Dataset):
    """Dataset for CSM-1B voice fine-tuning."""
    
    def __init__(self, data_dir: str, sample_rate: int = 16000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.samples = []
        
        # Load all segment files
        segments_dir = self.data_dir / "segments"
        if segments_dir.exists():
            for wav_file in sorted(segments_dir.glob("*.wav")):
                self.samples.append({
                    "path": str(wav_file),
                    "speaker": "israel",
                })
        
        # Also check for raw WAV files
        for wav_file in sorted(self.data_dir.glob("*.wav")):
            if "raw" not in wav_file.name:
                self.samples.append({
                    "path": str(wav_file),
                    "speaker": "israel",
                })
        
        log.info(f"Loaded {len(self.samples)} voice samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["path"])
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return {
            "audio": waveform.squeeze(0),
            "speaker_id": sample["speaker"],
            "duration": waveform.shape[0] / self.sample_rate,
        }


class CSMFineTuner:
    """Fine-tunes CSM-1B on custom voice data."""
    
    def __init__(self, model_name: str = "sesame/csm-1b", device: str = "cuda"):
        self.device = device
        log.info(f"Loading CSM-1B from {model_name}...")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        log.info(f"Model loaded on {device}")
        log.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B")
    
    def train(self, dataset: VoiceDataset, output_dir: str, epochs: int = 25,
              batch_size: int = 4, learning_rate: float = 1e-5):
        """Fine-tune the model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        log.info(f"Starting fine-tuning: {epochs} epochs, {len(dataloader)} batches/epoch")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Forward pass (simplified — actual CSM uses Mimi tokenization)
                    loss = self._compute_loss(batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        log.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        log.warning("OOM — skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    raise
            
            avg_loss = total_loss / len(dataloader)
            log.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                ckpt_path = output_path / f"checkpoint-epoch-{epoch+1}"
                self.model.save_pretrained(ckpt_path)
                log.info(f"Checkpoint saved: {ckpt_path}")
        
        # Save final model
        self.model.save_pretrained(output_path / "final")
        log.info(f"Fine-tuning complete. Model saved to {output_path / 'final'}")
        
        return output_path / "final"
    
    def _compute_loss(self, batch):
        """Compute loss for a batch (placeholder — actual CSM loss is more complex)."""
        # This is a simplified placeholder
        # Actual CSM training requires:
        # 1. Tokenize audio with Mimi → RVQ codes
        # 2. Tokenize text with Llama tokenizer
        # 3. Create interleaved sequence
        # 4. Cross-entropy on audio codes
        # The full implementation requires Sesame's Mimi tokenizer
        return torch.tensor(1.0, requires_grad=True, device=self.device)
    
    def generate_voice(self, text: str, output_path: str, speaker_id: str = "israel"):
        """Generate speech from text using fine-tuned model."""
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize text with speaker ID
            input_text = f"[{speaker_id}] {text}"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Generate (simplified)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
            )
            
            # Decode and save
            # Note: Actual CSM generates audio codes, not text
            # The audio codes need to be decoded through Mimi decoder
            generated = self.tokenizer.decode(outputs[0])
            log.info(f"Generated: {generated[:100]}...")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM-1B on custom voice")
    parser.add_argument("--data-dir", required=True, help="Directory with voice data")
    parser.add_argument("--output-dir", default="./csm-israel", help="Output directory")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just setup")
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("CSM-1B Voice Fine-Tuning Pipeline")
    log.info("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info(f"GPU: {gpu}, VRAM: {vram:.1f}GB")
        
        if vram < 16:
            log.warning(f"Low VRAM ({vram:.1f}GB). May need gradient checkpointing.")
    else:
        log.warning("No CUDA GPU found. Training will be very slow on CPU.")
    
    # Load dataset
    dataset = VoiceDataset(args.data_dir)
    
    if len(dataset) == 0:
        log.error(f"No voice samples found in {args.data_dir}")
        log.info("Run youtube_voice_extract.py first to extract voice data.")
        sys.exit(1)
    
    log.info(f"Total training samples: {len(dataset)}")
    total_dur = sum(s["duration"] for s in dataset.samples)
    log.info(f"Total audio duration: {total_dur:.1f}s ({total_dur/60:.1f}min)")
    
    # Checkpoint
    checkpoint_path = Path(args.output_dir) / "final"
    if checkpoint_path.exists():
        log.warning(f"Output already exists: {checkpoint_path}")
        log.info("Delete it or use a different --output-dir to retrain.")
        if not args.skip_train:
            args.skip_train = True
    
    if args.skip_train:
        log.info("Skipping training (--skip-train). Setup complete.")
        return
    
    # Initialize fine-tuner
    fine_tuner = CSMFineTuner()
    
    # Train
    # NOTE: This is a simplified placeholder. The actual CSM-1B fine-tuning
    # requires the Mimi tokenizer from moshi-ml and Sesame's training code.
    # For production fine-tuning, use the Speechmatics pipeline:
    # https://blog.speechmatics.com/sesame-finetune
    
    log.info("=" * 60)
    log.info("NOTE: This is a simplified template. For full CSM-1B fine-tuning:")
    log.info("1. Install moshi-ml: pip install moshi-ml")
    log.info("2. Download Mimi weights from HuggingFace")
    log.info("3. Use Speechmatics fine-tuning pipeline")
    log.info("4. Or use Unsloth's Sesame CSM notebook:")
    log.info("   https://colab.research.google.com/github/unslothai/notebooks/")
    log.info("=" * 60)
    
    log.info("Setup complete. Ready for fine-tuning.")


if __name__ == "__main__":
    main()
