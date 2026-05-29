"""
Outbound Sales Call Center — Powered by Microsoft VibeVoice
Orchestrates AI voice agents for outbound sales calls.

Architecture:
  Campaign Manager → Lead Queue → Voice Agent (VibeVoice) → Call Results
                                         ↕
                              VibeVoice-Realtime-0.5B (streaming TTS)
                              WebSocket telephony bridge

Hardware: MC (RTX 3070 8GB) for VibeVoice inference
          Sandbox for campaign orchestration + lead management
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ── Configuration ──────────────────────────────────────────────────────────

VIBEVOICE_API_URL = os.getenv("VIBEVOICE_API_URL", "http://192.168.1.212:9000")
VIBEVOICE_MODEL = os.getenv("VIBEVOICE_MODEL", "realtime-0.5b")
CAMPAIGN_MAX_CONCURRENT = int(os.getenv("CAMPAIGN_MAX_CONCURRENT", "5"))
CALL_TIMEOUT_SECONDS = int(os.getenv("CALL_TIMEOUT_SECONDS", "300"))
RESULTS_DIR = os.getenv("CALL_RESULTS_DIR", "/home/user/hermes-workspace/voice-agent-platform/call_results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data Models ────────────────────────────────────────────────────────────

class CallStatus(Enum):
    PENDING = "pending"
    DIALING = "dialing"
    CONNECTED = "connected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    VOICEMAIL = "voicemail"


class Disposition(Enum):
    NOT_INTERESTED = "not_interested"
    CALLBACK = "callback"
    MEETING_BOOKED = "meeting_booked"
    QUALIFIED = "lead_qualified"
    UNQUALIFIED = "lead_unqualified"
    DO_NOT_CALL = "dnc"
    VOICEMAIL_LEFT = "voicemail_left"


@dataclass
class Lead:
    id: str
    name: str
    phone: str
    company: str = ""
    email: str = ""
    industry: str = ""
    notes: str = ""
    priority: int = 0  # Higher = more important
    status: CallStatus = CallStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    best_time: str = ""  # e.g., "morning", "afternoon"
    custom_prompt_vars: dict = field(default_factory=dict)


@dataclass
class CallResult:
    call_id: str
    lead_id: str
    status: CallStatus
    disposition: Optional[Disposition] = None
    duration_seconds: float = 0.0
    transcript: list = field(default_factory=list)
    ai_summary: str = ""
    interest_score: float = 0.0  # 0-1
    audio_file: str = ""
    timestamp: float = field(default_factory=time.time)
    error: str = ""


@dataclass
class Campaign:
    id: str
    name: str
    script: str
    voice_id: str = "default"
    max_calls: int = 100
    max_concurrent: int = 5
    leads: list = field(default_factory=list)
    results: list = field(default_factory=list)
    active: bool = False


# ── Voice Agent (VibeVoice Bridge) ─────────────────────────────────────────

class VibeVoiceAgent:
    """Bridges to VibeVoice-Realtime-0.5B for streaming TTS."""

    def __init__(self, api_url: str = VIBEVOICE_API_URL):
        self.api_url = api_url
        self.model_loaded = False

    async def healthcheck(self) -> bool:
        """Check if VibeVoice server is reachable."""
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.api_url}/health", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def speak(self, text: str, voice: str = "default") -> bytes:
        """Generate speech audio from text via VibeVoice."""
        import urllib.request
        payload = json.dumps({
            "text": text,
            "voice": voice,
            "stream": False
        }).encode()
        req = urllib.request.Request(
            f"{self.api_url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    async def speak_streaming(self, text: str, voice: str = "default"):
        """Stream speech audio chunks via WebSocket for real-time calls."""
        import urllib.request
        payload = json.dumps({
            "text": text,
            "voice": voice,
            "stream": True
        }).encode()
        req = urllib.request.Request(
            f"{self.api_url}/synthesize_stream",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        # Generator yields audio chunks
        with urllib.request.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                yield chunk


# ── Conversation Manager ───────────────────────────────────────────────────

class ConversationManager:
    """Manages the dialogue flow during an outbound call."""

    # Sales script template with variable substitution
    OPENING = """
    Hello {name}, this is {agent_name} calling from {company}. 
    {hook}
    Do you have a quick moment to chat?
    """

    VALUE_PROPS = {
        "default": "We help {industry} companies increase revenue by 30% through AI automation.",
        "ecommerce": "We help ecommerce brands recover 20% more abandoned carts using AI voice agents.",
        "saas": "We help SaaS companies reduce churn by 35% with predictive AI outreach.",
        "realestate": "We help real estate firms book 3x more appointments with AI-powered lead follow-up.",
        "insurance": "We help insurance agencies pre-qualify leads in real-time, cutting cost-per-acquisition by 40%.",
    }

    OBJECTION_HANDLERS = {
        "not_interested": "I completely understand. Would it be okay if I sent you a brief overview by email? You can review it on your own time.",
        "too_busy": "I respect your time — this will take just 60 seconds. If it's not relevant, I'll hang up immediately. Fair enough?",
        "already_have": "That's great — what solution are you currently using? I'd love to share a quick comparison, no pressure.",
        "send_info": "Absolutely. What's the best email to send that to? I'll include a calendar link so you can book a quick demo if you're interested.",
        "callback": "Of course. When works best for you? Morning or afternoon? I'll note it in our system.",
        "cost": "That's a fair question. Most of our clients see ROI within the first 30 days. Would a quick 10-minute walkthrough help clarify the investment?",
    }

    CLOSING = """
    Thank you for your time, {name}. {closing_action}
    Have a great {time_of_day}!
    """

    def __init__(self, lead: Lead, script_variant: str = "default"):
        self.lead = lead
        self.script_variant = script_variant
        self.turn_count = 0
        self.interest_signals = 0
        self.objection_count = 0
        self.transcript = []

    def get_opening(self, agent_name: str = "Alex", company: str = "E5 Enclave") -> str:
        """Generate the opening line with lead-specific personalization."""
        hook = self.VALUE_PROPS.get(
            self.lead.industry.lower(),
            self.VALUE_PROPS["default"]
        ).format(industry=self.lead.industry or "your")

        opening = self.OPENING.format(
            name=self.lead.name.split()[0],
            agent_name=agent_name,
            company=company,
            hook=hook
        )
        self._log_turn("agent", opening)
        return opening

    def handle_response(self, prospect_said: str) -> tuple[str, bool]:
        """
        Process prospect's response and generate AI reply.
        Returns: (reply_text, should_continue)
        """
        self._log_turn("prospect", prospect_said)
        self.turn_count += 1

        # Analyze sentiment/interest
        interest_keywords = ["yes", "interested", "tell me more", "how", "what", "sure", "okay", "book", "demo", "meeting", "price", "cost"]
        objection_keywords = ["no", "not interested", "stop", "busy", "already", "goodbye", "don't call", "remove"]
        dnc_keywords = ["do not call", "dnc", "remove my number", "stop calling", "unsubscribe"]

        text_lower = prospect_said.lower()

        # Check DNC first
        if any(kw in text_lower for kw in dnc_keywords):
            reply = "I understand and apologize for the inconvenience. I'll make sure you're removed from our list immediately. Have a good day."
            self._log_turn("agent", reply)
            return reply, False

        # Check objections
        if any(kw in text_lower for kw in objection_keywords):
            self.objection_count += 1
            handler_key = self._match_objection(text_lower)
            reply = self.OBJECTION_HANDLERS.get(handler_key, self.OBJECTION_HANDLERS["send_info"])
            self._log_turn("agent", reply)
            return reply, self.objection_count < 2

        # Check interest
        if any(kw in text_lower for kw in interest_keywords):
            self.interest_signals += 1
            if self.interest_signals >= 2:
                reply = self._generate_closing_ask()
            else:
                reply = self._generate_value_prop()
            self._log_turn("agent", reply)
            return reply, True

        # Neutral/general response
        reply = "I appreciate that. Let me ask — what's your biggest challenge with {pain_point} right now?".format(
            pain_point=self.lead.industry or "lead generation"
        )
        self._log_turn("agent", reply)
        return reply, self.turn_count < 8

    def _match_objection(self, text: str) -> str:
        if "busy" in text or "time" in text:
            return "too_busy"
        if "already" in text:
            return "already_have"
        if "send" in text or "email" in text:
            return "send_info"
        if "cost" in text or "price" in text or "much" in text:
            return "cost"
        if "call back" in text or "callback" in text:
            return "callback"
        return "not_interested"

    def _generate_value_prop(self) -> str:
        props = self.VALUE_PROPS.get(self.script_variant, self.VALUE_PROPS["default"])
        return f"Here's the thing — {props} The whole process takes about 10 minutes to set up. Can I walk you through a quick example?"

    def _generate_closing_ask(self) -> str:
        return "It sounds like this could be a great fit. Would you be open to a quick 15-minute demo this week? I can send a calendar link right now."

    def get_disposition(self) -> Disposition:
        """Determine final disposition from conversation."""
        if self.interest_signals >= 3:
            return Disposition.QUALIFIED
        elif self.interest_signals >= 2:
            return Disposition.CALLBACK
        elif self.objection_count >= 2:
            return Disposition.NOT_INTERESTED
        else:
            return Disposition.UNQUALIFIED

    def get_interest_score(self) -> float:
        """Calculate interest score 0-1."""
        base = self.interest_signals * 0.25
        penalty = self.objection_count * 0.15
        turn_bonus = min(self.turn_count * 0.05, 0.2)
        return max(0.0, min(1.0, base - penalty + turn_bonus))

    def _log_turn(self, speaker: str, text: str):
        self.transcript.append({
            "turn": self.turn_count,
            "speaker": speaker,
            "text": text.strip(),
            "timestamp": time.time()
        })


# ── Call Center Orchestrator ───────────────────────────────────────────────

class OutboundCallCenter:
    """Main orchestrator for outbound sales campaigns."""

    def __init__(self):
        self.voice_agent = VibeVoiceAgent()
        self.active_calls: dict[str, asyncio.Task] = {}
        self.campaigns: dict[str, Campaign] = {}
        self.results: list[CallResult] = []

    def create_campaign(self, name: str, script: str, leads: list[Lead],
                        voice_id: str = "default", max_calls: int = 100) -> Campaign:
        """Create a new outbound campaign."""
        campaign = Campaign(
            id=str(uuid.uuid4())[:8],
            name=name,
            script=script,
            voice_id=voice_id,
            max_calls=max_calls,
            leads=leads
        )
        self.campaigns[campaign.id] = campaign
        return campaign

    async def run_campaign(self, campaign_id: str):
        """Execute a campaign — process all leads concurrently up to max_concurrent."""
        campaign = self.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign.active = True
        semaphore = asyncio.Semaphore(campaign.max_concurrent)
        processed = 0

        print(f"[Campaign:{campaign.name}] Starting — {len(campaign.leads)} leads, max {campaign.max_concurrent} concurrent")

        async def _call_with_limit(lead: Lead):
            nonlocal processed
            async with semaphore:
                if processed >= campaign.max_calls:
                    return
                result = await self._execute_call(lead, campaign)
                campaign.results.append(result)
                self.results.append(result)
                processed += 1

        tasks = [_call_with_limit(lead) for lead in campaign.leads[:campaign.max_calls]]
        await asyncio.gather(*tasks, return_exceptions=True)

        campaign.active = False
        self._save_results(campaign)
        return campaign.results

    async def _execute_call(self, lead: Lead, campaign: Campaign) -> CallResult:
        """Execute a single outbound call."""
        call_id = str(uuid.uuid4())[:12]
        lead.attempts += 1
        lead.status = CallStatus.DIALING

        start_time = time.time()
        conversation = ConversationManager(lead)
        result = CallResult(call_id=call_id, lead_id=lead.id, status=CallStatus.DIALING)

        try:
            # Phase 1: Opening
            opening = conversation.get_opening()
            audio = await self.voice_agent.speak(opening, campaign.voice_id)
            lead.status = CallStatus.CONNECTED

            # Phase 2: Conversation loop (simulated for now — telephony bridge would drive this)
            # In production, this would be a WebSocket loop with Twilio/Vonage
            # For now, we generate the full conversation audio
            full_script = opening

            # Phase 3: Generate full call audio
            result.status = CallStatus.COMPLETED
            result.disposition = conversation.get_disposition()
            result.interest_score = conversation.get_interest_score()
            result.transcript = conversation.transcript
            result.duration_seconds = time.time() - start_time

            # Save audio
            audio_path = os.path.join(RESULTS_DIR, f"{call_id}.wav")
            with open(audio_path, "wb") as f:
                f.write(audio)
            result.audio_file = audio_path

            lead.status = CallStatus.COMPLETED
            print(f"  [Call:{call_id}] {lead.name} → {result.disposition.value} (score: {result.interest_score:.1%})")

        except Exception as e:
            result.status = CallStatus.FAILED
            result.error = str(e)
            lead.status = CallStatus.FAILED
            print(f"  [Call:{call_id}] {lead.name} → ERROR: {e}")

        return result

    def _save_results(self, campaign: Campaign):
        """Persist campaign results to JSON."""
        results_file = os.path.join(RESULTS_DIR, f"campaign_{campaign.id}_{int(time.time())}.json")
        data = {
            "campaign": {
                "id": campaign.id,
                "name": campaign.name,
                "voice_id": campaign.voice_id,
                "total_leads": len(campaign.leads),
            },
            "summary": {
                "total_calls": len(campaign.results),
                "completed": sum(1 for r in campaign.results if r.status == CallStatus.COMPLETED),
                "failed": sum(1 for r in campaign.results if r.status == CallStatus.FAILED),
                "qualified": sum(1 for r in campaign.results if r.disposition == Disposition.QUALIFIED),
                "callbacks": sum(1 for r in campaign.results if r.disposition == Disposition.CALLBACK),
                "avg_interest": sum(r.interest_score for r in campaign.results) / max(len(campaign.results), 1),
            },
            "results": [
                {
                    "call_id": r.call_id,
                    "lead_id": r.lead_id,
                    "status": r.status.value,
                    "disposition": r.disposition.value if r.disposition else None,
                    "interest_score": r.interest_score,
                    "duration": r.duration_seconds,
                }
                for r in campaign.results
            ]
        }
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Campaign:{campaign.name}] Results saved → {results_file}")

    def get_stats(self) -> dict:
        """Return aggregate stats across all campaigns."""
        total = len(self.results)
        if total == 0:
            return {"total_calls": 0}
        return {
            "total_calls": total,
            "completed": sum(1 for r in self.results if r.status == CallStatus.COMPLETED),
            "failed": sum(1 for r in self.results if r.status == CallStatus.FAILED),
            "qualified": sum(1 for r in self.results if r.disposition == Disposition.QUALIFIED),
            "callbacks": sum(1 for r in self.results if r.disposition == Disposition.CALLBACK),
            "dnc": sum(1 for r in self.results if r.disposition == Disposition.DO_NOT_CALL),
            "avg_interest_score": sum(r.interest_score for r in self.results) / total,
            "avg_duration": sum(r.duration_seconds for r in self.results) / total,
        }


# ── Voice Cloning for Israel's Sovereign Voice ──────────────────────────────

class SovereignVoiceCloner:
    """
    Fine-tune VibeVoice on Israel's voice samples.
    Uses Israel's recorded sales calls / voice memos as training data.

    Training pipeline:
    1. Collect voice samples (Whisper-transcribed)
    2. Create paired (text, audio) dataset
    3. Fine-tune VibeVoice-Realtime-0.5B using LoRA
    4. Deploy as custom voice in call center
    """

    def __init__(self, voice_samples_dir: str = "/home/user/hermes-workspace/voice-agent-platform/voice_samples"):
        self.samples_dir = voice_samples_dir
        self.model_path = "microsoft/VibeVoice-Realtime-0.5B"

    def prepare_dataset(self, audio_files: list[str]) -> list[dict]:
        """Prepare (text, audio) pairs from voice samples."""
        dataset = []
        for audio_file in audio_files:
            # Whisper would transcribe — placeholder
            dataset.append({
                "audio": audio_file,
                "text": "",  # Whisper fills this
                "speaker": "israel"
            })
        return dataset

    def get_training_config(self) -> dict:
        """Return LoRA fine-tuning config for sovereign voice."""
        return {
            "model": self.model_path,
            "method": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "max_new_tokens": 250,
            "batch_size": 2,
            "epochs": 10,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "save_steps": 200,
            "fp16": False,
            "bf16": True,
            "output_dir": "/home/user/hermes-workspace/voice-agent-platform/sovereign-voice-lora",
            "wandb_mode": "disabled",
        }


# ── Entrypoint ──────────────────────────────────────────────────────────────

async def main():
    """Demo: Create and run a small outbound campaign."""
    center = OutboundCallCenter()

    # Sample leads
    leads = [
        Lead(id="L001", name="John Smith", phone="+15551234567", company="Acme Corp", industry="saas"),
        Lead(id="L002", name="Sarah Johnson", phone="+15552345678", company="TechFlow", industry="ecommerce"),
        Lead(id="L003", name="Mike Chen", phone="+15553456789", company="DataPrime", industry="saas"),
    ]

    campaign = center.create_campaign(
        name="E5 Enclave Outbound v1",
        script="default",
        leads=leads,
        voice_id="default",
        max_calls=10
    )

    print(f"Campaign created: {campaign.id} — '{campaign.name}'")
    print(f"Leads: {len(leads)}")
    print(f"VibeVoice endpoint: {VIBEVOICE_API_URL}")
    print(f"Results dir: {RESULTS_DIR}")
    print()
    print("NOTE: Full call execution requires VibeVoice server running on MC.")
    print("      Deploy with: python vibevoice_server.py (on MC)")
    print("      Then set VIBEVOICE_API_URL=http://192.168.1.212:9000")


if __name__ == "__main__":
    asyncio.run(main())
