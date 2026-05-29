"""
Telephony Bridge — Twilio Integration for VibeVoice Call Center
Handles inbound/outbound call routing, WebSocket streaming to VibeVoice.
"""

import os
import json
import asyncio
import base64
import hashlib
import hmac
import time
from typing import Optional
from urllib.parse import urlencode

# ── Twilio Config ──────────────────────────────────────────────────────────

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
VIBEVOICE_WS_URL = os.getenv("VIBEVOICE_WS_URL", "ws://192.168.1.212:9000/synthesize_stream")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL", "https://your-domain.com/twilio")


# ── Twilio Signature Verification ──────────────────────────────────────────



# ── Twilio Signature Verification ──────────────────────────────────────────

def validate_twilio_request(request_body: bytes, signature: str, url: str) -> bool:
    """Validate that request actually came from Twilio."""
    if not TWILIO_AUTH_TOKEN:
        return False  # Can't validate without auth token
    
    # Twilio signs: URL + sorted params
    parsed = url + request_body.decode("utf-8")
    expected = base64.b64encode(
        hmac.new(
            TWILIO_AUTH_TOKEN.encode(),
            parsed.encode(),
            hashlib.sha1
        ).digest()
    ).decode()
    return hmac.compare_digest(expected, signature)


# ── TwiML Generation ──────────────────────────────────────────────────────

def generate_vibevoice_twiml(ws_url: str = TWILIO_WS_URL) -> str:
    """
    Generate TwiML that connects a phone call to VibeVoice WebSocket.
    Twilio streams raw audio <-> VibeVoice TTS/ASR.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{ws_url}" track="both_tracks" />
    </Start>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
    <Say voice="Polly.Joanna">Please wait while we connect you.</Say>
    <Pause length="60"/>
</Response>"""


def generate_outbound_twiml(target_number: str) -> str:
    """Generate TwiML for outbound call."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial answerOnBridge="true" callerId="{TWILIO_PHONE_NUMBER}">
        <Number>{target_number}</Number>
    </Dial>
</Response>"""


def generate_voicemail_twiml(audio_url: str) -> str:
    """Generate TwiML for voicemail drop."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Pause length="2"/>
    <Play>{audio_url}</Play>
</Response>"""


# ── Twilio API Client ──────────────────────────────────────────────────────

class TwilioClient:
    """Lightweight Twilio REST client (no external dependency)."""
    
    BASE = "https://api.twilio.com/2010-04-01"
    
    def __init__(self, account_sid: str = TWILIO_ACCOUNT_SID, auth_token: str = TWILIO_AUTH_TOKEN):
        self.sid = account_sid
        self.token = auth_token
    
    def _request(self, method: str, path: str, data: dict = None) -> dict:
        """Make authenticated request to Twilio API."""
        import urllib.request
        url = f"{self.BASE}/Accounts/{self.sid}/{path}"
        auth = base64.b64encode(f"{self.sid}:{self.token}".encode()).decode()
        
        req = urllib.request.Request(
            url,
            data=urlencode(data).encode() if data else None,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": str(e)}
    
    def make_call(self, to: str, from_: str = TWILIO_PHONE_NUMBER, 
                  webhook_url: str = TWILIO_WEBHOOK_URL) -> dict:
        """Initiate an outbound call."""
        return self._request("POST", "Calls.json", {
            "To": to,
            "From": from_,
            "Url": f"{webhook_url}/twiml/outbound",
            "StatusCallback": f"{webhook_url}/status",
            "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
            "MachineDetection": "Enable",
            "AsyncAmd": "true",
            "AsyncAmdStatusCallback": f"{webhook_url}/amd",
        })
    
    def send_calls_batch(self, numbers: list[str], webhook_url: str = TWILIO_WEBHOOK_URL) -> list[dict]:
        """Fire off a batch of outbound calls."""
        results = []
        for number in numbers:
            result = self.make_call(number, webhook_url=webhook_url)
            results.append(result)
            # Rate limit: max 1 call/second for most Twilio accounts
            time.sleep(1.1)
        return results
    
    def get_call(self, call_sid: str) -> dict:
        """Get call status/details."""
        return self._request("GET", f"Calls/{call_sid}.json")
    
    def get_recordings(self, call_sid: str) -> list:
        """Get recordings for a call."""
        result = self._request("GET", f"Calls/{call_sid}/Recordings.json")
        return result.get("recordings", [])
    
    def list_calls(self, status: str = None, limit: int = 50) -> list:
        """List recent calls."""
        params = {"PageSize": str(limit)}
        if status:
            params["Status"] = status
        qs = urlencode(params)
        result = self._request("GET", f"Calls.json?{qs}")
        return result.get("calls", [])
    
    def buy_number(self, area_code: str = None, contains: str = None) -> dict:
        """Search and buy a phone number."""
        import urllib.request
        search_params = {"Capability": ["voice"]}
        if area_code:
            search_params["AreaCode"] = area_code
        
        # Search available numbers
        url = f"{self.BASE}/Accounts/{self.sid}/AvailablePhoneNumbers/US/Local.json"
        req = urllib.request.Request(
            url + "?" + urlencode({"AreaCode": area_code} if area_code else {}),
            headers={"Authorization": f"Basic {base64.b64encode(f'{self.sid}:{self.token}'.encode()).decode()}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        
        numbers = data.get("available_phone_numbers", [])
        if not numbers:
            return {"error": "No available numbers"}
        
        # Buy the first one
        return self._request("POST", "IncomingPhoneNumbers.json", {
            "PhoneNumber": numbers[0]["phone_number"],
        })
    
    def get_usage(self, start_date: str = None, end_date: str = None) -> list:
        """Get usage/cost data."""
        params = "&".join(filter(None, []))
        result = self._request("GET", f"Usage/Records.json?{params}" if params else "Usage/Records.json")
        return result.get("usage_records", [])


# ── Call Audio Pipeline: Twilio ↔ VibeVoice ────────────────────────────────

class CallAudioPipeline:
    """
    Bridges Twilio Media Streams to VibeVoice WebSocket.
    
    Flow:
    1. Twilio call connects → WebSocket opens to this pipeline
    2. Twilio sends raw audio (mulaw/8kHz) → Forward to VibeVoice-ASR
    3. VibeVoice-ASR transcribes → Generate response
    4. VibeVoice-TTS synthesizes response → Send back to Twilio
    5. Loop until call ends
    """
    
    def __init__(self, vibevoice_ws_url: str = VIBEVOICE_WS_URL):
        self.vibevoice_ws = vibevoice_ws_url
        self.active_calls = {}
    
    async def handle_twilio_stream(self, websocket):
        """
        Handle Twilio bidirectional media stream.
        Twilio sends JSON messages with base64 audio chunks.
        """
        import websockets
        
        call_sid = None
        stream_sid = None
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                event = data.get("event")
                
                if event == "connected":
                    # Twilio stream connected
                    protocol = data.get("protocol", "")
                    print(f"[Twilio] Stream connected: {protocol}")
                
                elif event == "start":
                    # Call started, extract metadata
                    call_sid = data.get("start", {}).get("callSid")
                    stream_sid = data.get("start", {}).get("streamSid")
                    print(f"[Twilio] Call started: {call_sid}")
                    
                    # Connect to VibeVoice for this call
                    await self._connect_vibevoice(call_sid, stream_sid)
                
                elif event == "media":
                    # Audio from Twilio (base64 mulaw 8kHz)
                    audio_payload = data.get("media", {}).get("payload", "")
                    
                    if audio_payload and call_sid:
                        # Decode mulaw → forward to VibeVoice ASR
                        # VibeVoice will transcribe + generate response
                        response_audio = await self._process_audio(
                            call_sid, audio_payload
                        )
                        
                        if response_audio:
                            # Send back to Twilio
                            await websocket.send(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": response_audio
                                }
                            }))
                
                elif event == "stop":
                    print(f"[Twilio] Call ended: {call_sid}")
                    self.active_calls.pop(call_sid, None)
                    break
        
        except Exception as e:
            print(f"[Twilio] Stream error: {e}")
            if call_sid:
                self.active_calls.pop(call_sid, None)
    
    async def _connect_vibevoice(self, call_sid: str, stream_sid: str):
        """Establish VibeVoice connection for a call."""
        self.active_calls[call_sid] = {
            "stream_sid": stream_sid,
            "start_time": time.time(),
            "turns": 0,
        }
    
    async def _process_audio(self, call_sid: str, mulaw_audio: str) -> Optional[str]:
        """
        Process audio: VibeVoice ASR → Conversation → VibeVoice TTS.
        Returns base64 mulaw audio response.
        """
        call_info = self.active_calls.get(call_sid)
        if not call_info:
            return None
        
        call_info["turns"] += 1
        
        # In production:
        # 1. Decode mulaw base64 → raw bytes
        # 2. Send to VibeVoice ASR → get transcript
        # 3. Run through ConversationManager → get response text
        # 4. Send text to VibeVoice TTS → get audio
        # 5. Encode audio as mulaw base64 → return
        
        return None  # Placeholder — actual audio processing in production


# ── Voicemail Drop Engine ──────────────────────────────────────────────────

class VoicemailDrop:
    """
    Pre-records voicemails and drops them when call goes to VM.
    Uses Twilio's machine detection + async AMD.
    """
    
    TEMPLATES = {
        "intro": "Hi {name}, this is {agent} from {company}. {message}",
        "callback": "{name}, this is {agent} from {company}. I'd love to show you how we help {industry} companies. Call me back at {number}.",
        "follow_up": "Hey {name}, just following up on my previous call. {message} Reach me at {number}.",
        "meeting": "{name}, wanted to send you that calendar link I mentioned. {link}. Looking forward to connecting!",
    }
    
    def __init__(self, twilio: TwilioClient):
        self.twilio = twilio
    
    def generate_voicemail_text(self, lead, template: str = "callback", **kwargs) -> str:
        """Generate personalized voicemail script."""
        tmpl = self.TEMPLATES.get(template, self.TEMPLATES["callback"])
        return tmpl.format(
            name=lead.name.split()[0] if lead.name else "there",
            agent=kwargs.get("agent", "Alex"),
            company=kwargs.get("company", "E5 Enclave"),
            industry=lead.industry or "your industry",
            number=TWILIO_PHONE_NUMBER,
            message=kwargs.get("message", ""),
            link=kwargs.get("link", ""),
        )
    
    async def generate_voicemail_audio(self, text: str, voice: str = "alloy") -> bytes:
        """Generate voicemail audio using VibeVoice."""
        import urllib.request
        payload = json.dumps({"text": text, "voice": voice}).encode()
        req = urllib.request.Request(
            f"{VIBEVOICE_WS_URL}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
