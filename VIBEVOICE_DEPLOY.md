# VibeVoice Deployment Playbook
## Outbound Sales Call Center — Sovereign Voice Architecture

### Phase 1: Deploy VibeVoice on MC (RTX 3070)

```powershell
# On Monte-Cristo (Windows PowerShell as Admin)

# 1. Install Python 3.11+ if not present
winget install Python.Python.3.11

# 2. Create VibeVoice venv
python -m venv C:\vibevoice\venv
C:\vibevoice\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors
pip install fastapi uvicorn websockets
pip install bitsandbytes  # For int8 quantization

# 4. Clone the server
git clone https://github.com/microsoft/VibeVoice.git C:\vibevoice\source

# 5. Copy our server wrapper
# (SCP or copy vibevoice_server.py from sandbox)

# 6. Start the Realtime-0.5B server (for streaming calls)
python vibevoice_server.py --model realtime-0.5b --port 9000

# 7. In another terminal: Start the TTS-1.5B server (for batch generation)
python vibevoice_server.py --model 1.5b --port 9002 --quantize int8

# 8. Verify
curl http://localhost:9000/health
curl http://localhost:9002/health
```

### Phase 2: Install Telephony Bridge

Option A: Twilio (recommended for production)
- Sign up at twilio.com → Get a phone number ($1/mo)
- Install: pip install twilio
- Configure TwiML webhooks → VibeVoice streaming endpoint

Option B: Vonage (formerly Nexmo)
- Better EU coverage, competitive pricing
- pip install vonage

Option C: Plivo
- Cheapest for high volume
- pip install plivo

### Phase 3: Voice Cloning — Sovereign Voice

Fine-tune VibeVoice on Israel's voice:

1. Collect 30+ minutes of Israel's voice (sales recordings, voice memos)
2. Whisper transcribe with timestamps
3. Run LoRA fine-tune on MC:
```powershell
pip install peft datasets
python sovereign_voice_train.py --data voice_samples/ --output sovereign-lora/
```
4. Deploy: Load sovereign-lora adapter in vibevoice_server.py

### Phase 4: Campaign Launch

```python
from outbound_call_center import OutboundCallCenter, Lead

center = OutboundCallCenter()
leads = [
    Lead(name="...", phone="...", company="...", industry="..."),
    # ... load from CSV
]
campaign = center.create_campaign("Sovereign Surge Q1", leads=leads)
await center.run_campaign(campaign.id)
```

### Cost Estimates

| Component | Cost |
|-----------|------|
| Twilio phone number | $1/mo |
| Outbound calls | $0.013/min |
| VibeVoice (self-hosted) | Free (RTX 3070) |
| 1000 calls × 3min avg | ~$40 |
| Expected conversion 5-10% | 50-100 qualified leads |
| Cost per qualified lead | $0.40-0.80 |

### Compliance (TCPA)

- Scrub against DNC list before calling
- Include opt-out in every voicemail
- Honor DNC requests immediately
- Call only 8am-9pm local time
- Maintain call records for 4+ years
