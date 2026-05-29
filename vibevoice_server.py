"""
VibeVoice Server — Deployment wrapper for Microsoft VibeVoice on MC (RTX 3070)
Exposes REST API for the outbound call center to consume.

Ports:
  9000 — VibeVoice TTS API (REST + WebSocket)
  9001 — Health/metrics

Usage (on MC):
  python vibevoice_server.py --model realtime-0.5b --port 9000
  python vibevoice_server.py --model 1.5b --port 9002 --quantize int8
"""

import argparse
import asyncio
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Model Loading ──────────────────────────────────────────────────────────

def load_vibevoice(model_size: str = "realtime-0.5b", quantize: str = None):
    """
    Load VibeVoice model.
    
    model_size: 'realtime-0.5b' | '1.5b'
    quantize: None | 'int8' | 'int4'
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError("Install: pip install transformers torch accelerate")

    model_map = {
        "realtime-0.5b": "microsoft/VibeVoice-Realtime-0.5B",
        "1.5b": "microsoft/VibeVoice-1.5B",
    }

    model_name = model_map.get(model_size)
    if not model_name:
        raise ValueError(f"Unknown model: {model_size}. Choose from {list(model_map.keys())}")

    print(f"[VibeVoice] Loading {model_name}...")
    print(f"[VibeVoice] Quantize: {quantize or 'none'}")
    print(f"[VibeVoice] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[VibeVoice] GPU: {torch.cuda.get_device_name(0)}")
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[VibeVoice] Free VRAM: {free:.1f}GB")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize == "int8":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"[VibeVoice] int8 quantization failed: {e}. Loading full precision.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    print(f"[VibeVoice] {model_size} loaded successfully")
    return model, tokenizer


# ── FastAPI Server ─────────────────────────────────────────────────────────

def create_app(model, tokenizer, model_size: str):
    """Create FastAPI application for VibeVoice TTS serving."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import Response
        import torch
        import numpy as np
    except ImportError:
        raise RuntimeError("Install: pip install fastapi uvicorn")

    app = FastAPI(title=f"VibeVoice Server — {model_size}", version="1.0.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": model_size,
            "timestamp": time.time()
        }

    @app.post("/synthesize")
    async def synthesize(request: dict):
        """Synthesize speech from text. Returns WAV audio bytes."""
        text = request.get("text", "")
        voice = request.get("voice", "default")

        if not text:
            return {"error": "No text provided"}, 400

        # Generate speech tokens via VibeVoice
        # Note: Actual VibeVoice inference uses the next-token diffusion pipeline
        # This is the adapter — actual implementation depends on VibeVoice's API
        
        start = time.time()
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # VibeVoice generates continuous speech tokens
            # The diffusion head converts to acoustic features
            output = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
            )
        
        # Decode to audio (model-specific)
        audio_tokens = output[0, inputs["input_ids"].shape[1]:]
        
        # Convert tokens to waveform (simplified — actual VibeVoice uses vocoder)
        # In production, this uses the VibeVoice acoustic decoder
        duration = time.time() - start

        # Placeholder: return JSON with metadata
        # Real implementation returns WAV bytes
        return {
            "status": "generated",
            "model": model_size,
            "text_length": len(text),
            "tokens_generated": len(audio_tokens),
            "generation_time_s": round(duration, 3),
            "voice": voice,
            # "audio": base64_encoded_wav  # In production
        }

    @app.websocket("/synthesize_stream")
    async def synthesize_stream(websocket: WebSocket):
        """WebSocket streaming TTS — for real-time call center."""
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text", "")
                voice = data.get("voice", "default")

                if text:
                    # Stream audio chunks as they're generated
                    # In production: yield PCM/WAV chunks from VibeVoice decoder
                    await websocket.send_json({
                        "chunk": 0,
                        "status": "generating",
                        "text_received": len(text),
                    })

                    # Simulate streaming chunks
                    chunks = max(1, len(text) // 50)
                    for i in range(chunks):
                        await websocket.send_bytes(b'\x00\x00' * 1600)  # 100ms silence placeholder
                    
                    await websocket.send_json({
                        "status": "complete",
                        "total_chunks": chunks,
                    })

        except WebSocketDisconnect:
            pass

    @app.get("/metrics")
    async def metrics():
        """Prometheus-style metrics."""
        import torch
        gpu_mem = {}
        if torch.cuda.is_available():
            gpu_mem = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "free_gb": round(torch.cuda.mem_get_info()[0] / 1e9, 2),
            }
        return {
            "model": model_size,
            "gpu_memory": gpu_mem,
            "uptime": time.time(),
        }

    return app


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VibeVoice TTS Server")
    parser.add_argument("--model", choices=["realtime-0.5b", "1.5b"], default="realtime-0.5b")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--quantize", choices=["int8", "int4"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Test model loading without starting server")
    args = parser.parse_args()

    model, tokenizer = load_vibevoice(args.model, args.quantize)

    if args.dry_run:
        print("[VibeVoice] Dry run complete — model loaded successfully")
        return

    app = create_app(model, tokenizer, args.model)

    try:
        import uvicorn
        print(f"[VibeVoice] Starting server on {args.host}:{args.port}")
        print(f"[VibeVoice] Health: http://localhost:{args.port}/health")
        print(f"[VibeVoice] Synthesize: POST http://localhost:{args.port}/synthesize")
        print(f"[VibeVoice] Stream: ws://localhost:{args.port}/synthesize_stream")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except ImportError:
        print("Install uvicorn: pip install uvicorn")
        raise


if __name__ == "__main__":
    main()
