# External Dependencies

## CSM-1B Fine-Tuning
- Repo: https://github.com/knottwill/sesame-finetune (109 stars, 11 forks)
- Purpose: Full CSM-1B fine-tuning by modifying original weights (not LoRA)
- Requirements: PyTorch 2.4, torchaudio, moshi 0.2.2, transformers, wandb, optuna
- Clone: `git clone https://github.com/knottwill/sesame-finetune.git`
- CSM base: `git clone https://github.com/SesameAILabs/csm.git && git checkout 836f8865`

## Speechmatics Fine-Tuning Guide
- URL: https://blog.speechmatics.com/sesame-finetune
- Covers: Data prep, pre-tokenization, training pipeline, hyperparameter sweep

## Unsloth CSM TTS Notebook
- URL: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_(1B)-TTS.ipynb
- Free GPU via Colab, optimized for lower VRAM

## Speaker Diarization (for voice isolation)
- pyannote/speaker-diarization-3.1 — identifies "who spoke when"
- Needs 8GB+ VRAM for GPU processing
- CPU possible but very slow for 4+ hours of audio
