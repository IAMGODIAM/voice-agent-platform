# VOICE AGENT PLATFORM — FULL PDR & IMPLEMENTATION PLAN
## Project "ACORN" — Conversational Voice System for E5 Enclave

**Document Type:** Product Design Review + Technical Architecture + Implementation Plan  
**Classification:** Chairman Directive — Full Wyrmcore Protocol  
**Date:** 2026-05-28  
**Author:** Hermie (Visionary Layer)  
**Executor:** Sue (CEO, Base44 Operations)  
**Priority:** P0 — All Hands On Deck  
**GitHub:** github.com/IAMGODIAM/voice-agent-platform (to be created)

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Market Recon — What Sesame Built](#2-market-recon--what-sesame-built)
3. [Technical Archaeology — What's Under the Hood](#3-technical-archaeology--whats-under-the-hood)
4. [Can We Build It? — Feasibility Analysis](#4-can-we-build-it--feasibility-analysis)
5. [Architecture Design — ACORN Platform](#5-architecture-design--acorn-platform)
6. [Base44 Agent Spec — The Voice Agent Code](#6-base44-agent-spec--the-voice-agent-code)
7. [Cloudflare Deployment Strategy](#7-cloudflare-deployment-strategy)
8. [Compute Requirements & Hardware Plan](#8-compute-requirements--hardware-plan)
9. [Voice Data Pipeline — Fine-Tuning Israel's Voice](#9-voice-data-pipeline--fine-tuning-israels-voice)
10. [Full Implementation Roadmap](#10-full-implementation-roadmap)
11. [Risk Register & Unknown Unknowns](#11-risk-register--unknown-unknowns)
12. [Success Criteria & KPIs](#12-success-criteria--kpis)
13. [Situation Room — Team Assignments](#13-situation-room--team-assignments)

---

## 1. EXECUTIVE SUMMARY

**The Opportunity:** Sesame AI just launched their iOS app (May 27, 2026) with voice agents so realistic they cross the uncanny valley. Their open-source CSM-1B model is the foundation. We can build a voice agent platform for E5 that powers call centers, outreach (ACORN), or any voice-first interaction — with Israel's own voice as the agent.

**The Stack:**
- **Voice Engine:** Sesame CSM-1B (open source, fine-tunable on Base44 compute)
- **Hosting:** Cloudflare Workers AI + Pages (Business account = custom domains, global CDN, serverless GPU)
- **Agent Platform:** Base44 (Sue's existing infrastructure) for conversation logic, memory, orchestration
- **Memory:** DAIM (our DAG-Immutable Agent Memory from Sprint 1-2) for persistent per-caller memory
- **Data Pipeline:** Voice recording → tokenization → fine-tuning → deployment

**The Chairman's Use Cases:**
1. **Call Center** — AI agents that sound human, handle inbound/outbound calls
2. **ACORN Outreach** — Voice-first donor/member engagement with persistent memory
3. **General Voice Interface** — Any E5 application where speaking is better than typing

**Bottom Line:** This is buildable NOW. The open-source model exists. Our Cloudflare Business account can host inference. Base44 can run the agent logic. We need to fine-tune on Israel's voice data and wire it all together.

---

## 2. MARKET RECON — WHAT SESAME BUILT

### 2.1 Sesame's Product (Launched May 27, 2026)

From their blog post "Voice Your Curiosity":
- **iOS app** with 4 character agents: Maya, Miles, Simone, Charlie
- **"Curiosity Engine"** — agents that help you learn, discover, reflect through voice
- **Conversational intelligence** — timing, rhythm, parallel search while speaking
- **Memory per agent** — what you discuss with Miles stays with Miles
- **Incognito mode** — ephemeral conversations, nothing saved
- **Search cards, notes, texting, deep dives** — all voice-first
- **Coming 2027:** Intelligent eyewear integration

### 2.2 Key Differentiators
- **Latency optimized** — "shaving off every millisecond"
- **Human-like imperfections** — deliberately incorporated (breath, rhythm, pauses)
- **Parallel search while speaking** — agents think while talking, pivot mid-sentence
- **Character consistency** — distinct voice, personality, point of view per agent

### 2.3 Business Context
- Sesame raised **$250M**, valued at **$1B+** (Oct 2025)
- Founded by ex-Meta AI chief **Yann LeCun advised**
- Competing with: ElevenLabs, Play.ht, OpenAI TTS, Google WaveNet
- Their moat: **conversational latency + character consistency + open-source base model**

---

## 3. TECHNICAL ARCHAEOLOGY — WHAT'S UNDER THE HOOD

### 3.1 CSM-1B Architecture (Open Source Component)

From the Speechmatics fine-tuning guide and GitHub repo:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CSM-1B ARCHITECTURE                         │
│                                                                 │
│  Text Input ──→ Llama-3 Tokenizer ──→ [speaker_id] + text      │
│                                          │                      │
│  Audio Input ──→ Mimi Tokenizer ──→ RVQ Audio Codes (32 CB)    │
│                                          │                      │
│                    ┌─────────────────────┘                      │
│                    ▼                                             │
│            INTERLEAVED TOKENS                                    │
│     [text_token, audio_token, text_token, audio_token, ...]    │
│                    │                                             │
│                    ▼                                             │
│     ┌──────────────────────────────┐                            │
│     │   BACKBONE TRANSFORMER       │  1B parameters             │
│     │   (Llama-variant)            │                            │
│     │   - Causal attention mask    │                            │
│     │   - Position embeddings      │                            │
│     │   - Produces hidden state h  │                            │
│     └──────────┬───────────────────┘                            │
│                │                                                 │
│         ┌──────┴──────┐                                         │
│         ▼             ▼                                          │
│  ┌─────────────┐ ┌──────────────────────────────┐              │
│  │ Codebook 0  │ │     AUDIO DECODER             │              │
│  │ Head (linear)│ │     100M parameters          │              │
│  │             │ │     - Takes h + codebook emb  │              │
│  │ Predicts    │ │     - Predicts CB 1-31        │              │
│  │ semantic CB │ │     - Compute amortization    │              │
│  └─────────────┘ │       (train on 1/16 frames)  │              │
│                  └──────────────┬───────────────┘              │
│                                 │                               │
│                                 ▼                               │
│                  32 RVQ Audio Codes ──→ Mimi Decoder ──→ WAV    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Mimi Audio Tokenizer (The Secret Sauce)

Mimi is what makes Sesame's voice sound human:

```
RAW AUDIO (16kHz)
    │
    ▼
┌─────────────────────────────────────────────┐
│  NEURAL ENCODER                             │
│  CNN → Transformer → Continuous Latent       │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌─────────────┐  ┌─────────────────────────────┐
│ Plain VQ    │  │  Multi-Level RVQ             │
│ (1 CB)      │  │  (31 CBs)                    │
│             │  │                              │
│ Semantic    │  │  Residual quantization:      │
│ codebook    │  │  iteration 1: quantize input │
│ trained via │  │  iteration 2: quantize error │
│ WavLM       │  │  ... × 31 levels             │
│ cosine      │  │                              │
│ similarity  │  │  Captures acoustic detail    │
└──────┬──────┘  └──────────┬──────────────────┘
       │                     │
       └──────────┬──────────┘
                  │
         32 Total Codebooks (RVQ codes)
                  │
                  ▼
         CSM generates these codes
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  NEURAL DECODER                             │
│  Inverse quantization → Transformer → CNN   │
│  → Reconstructed audio waveform              │
└─────────────────────────────────────────────┘
```

### 3.3 Multi-Resolution RVQ (Split-RVQ)

The key innovation: split information across codebooks efficiently:
- **CB 0 (Semantic):** High-level meaning, speaker identity. 1 codebook trained with WavLM embedding similarity
- **CB 1-31 (Acoustic):** Fine acoustic detail via residual quantization. Each level quantizes the ERROR from the previous level

This is why Mimi preserves nuance better than standard RVQ or waveform codecs.

### 3.4 Why It Sounds Human

1. **Discrete tokenization** — Forces model to learn phonetic structure, not waveform noise
2. **32 codebooks at 16kHz** — ~500 codes/second, capturing micro-prosody
3. **Speaker conditioning** — `[speaker_id]` prepended to text grounds voice identity
4. **Conversational training** — Model trained on dialogue pairs (text → audio), not isolated TTS
5. **Deliberate imperfections** — Sesame intentionally doesn't over-smooth; retains breath, pauses

---

## 4. CAN WE BUILD IT? — FEASIBILITY ANALYSIS

### 4.1 Component Checklist

| Component | Status | Source | Cost |
|---|---|---|---|
| **CSM-1B base model** | ✅ Open source | `sesame/csm-1b` on HuggingFace | Free |
| **Mimi tokenizer** | ✅ Open source | `kmhf/moshi-music` | Free |
| **Fine-tuning code** | ✅ Available | Speechmatics guide + Unsloth notebooks | Free |
| **Inference engine** | ✅ Available | Cloudflare Workers AI (Business) | ~$0.0001/1K tokens |
| **Hosting** | ✅ Available | Cloudflare Pages + Workers | Included in Business |
| **Agent orchestration** | ✅ Available | Base44 (Sue's platform) | Already deployed |
| **Memory system** | ✅ Available | DAIM (our Sprint 1-2 work) | Already deployed |
| **Training data** | ⚠️ Need to create | Israel's voice recordings | Time + effort |
| **GPU compute** | ⚠️ Need access | Cloudflare Workers AI or Modal/Lambda | ~$2-5/hr for fine-tuning |

### 4.2 What We CAN'T Get (Yet)

- **Sesame's training data** — Their conversational speech dataset is proprietary
- **CSM-3B / CSM-8B** — Only 1B was open-sourced; production demos likely use larger models
- **Sesame's micro-phone array tech** — They have custom hardware for voice pickup

### 4.3 What We CAN Build CSM-1B + fine-tuning

- **Custom voice cloning** — Fine-tune on Israel's voice (need 30min-2hr of clean audio)
- **Multi-character system** — Different speakers via `[speaker_id]` conditioning
- **Conversational voice agents** — Full duplex voice conversation
- **Call center integration** — Voice in → agent processes → voice out
- **Persistent memory** — Per-caller DAIM memory (ties to our DAG memory work)
- **Web + mobile interface** — Cloudflare Pages + WebRTC voice

---

## 5. ARCHITECTURE DESIGN — ACORN PLATFORM

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACORN VOICE PLATFORM                         │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Voice In    │    │  Web/App UI  │    │   Admin Dashboard    │  │
│  │  (WebRTC/     │    │  (React/     │    │   (Cloudflare        │  │
│  │   Phone Line) │    │   Next.js)   │    │    Pages)            │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                        │              │
│         └───────────────────┼────────────────────────┘              │
│                             │                                       │
│                    ┌────────┴────────┐                              │
│                    │  API GATEWAY    │                              │
│                    │  (Cloudflare    │                              │
│                    │   Workers)      │                              │
│                    │  + AI Gateway   │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              │              │              │                        │
│     ┌────────┴───────┐ ┌───┴────┐ ┌──────┴────────┐               │
│     │  VOICE ENGINE  │ │ AGENT  │ │   MEMORY      │               │
│     │                │ │ LOGIC  │ │   (DAIM)      │               │
│     │  CSM-1B        │ │        │ │               │               │
│     │  fine-tuned    │ │ Base44 │ │  DAG-based    │               │
│     │  on Israel's   │ │ agents │ │  immutable    │               │
│     │  voice         │ │        │ │  memory       │               │
│     │                │ │ Sue's  │ │  per-caller   │               │
│     │  Hosted on:    │ │ 12     │ │               │               │
│     │  Cloudflare    │ │ agents │ │  PostgreSQL   │               │
│     │  Workers AI    │ │        │ │  + Neo4j      │               │
│     └────────┬───────┘ └───┬────┘ └──────┬────────┘               │
│              │             │              │                         │
│              └─────────────┼──────────────┘                       │
│                            │                                       │
│                   ┌────────┴────────┐                              │
│                   │   DATA LAYER    │                              │
│                   │                 │                              │
│                   │  D1 (metadata)  │                              │
│                   │  R2 (audio)     │                              │
│                   │  KV (sessions)  │                              │
│                   │  Vectorize      │                              │
│                   │  (embeddings)   │                              │
│                   └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Voice Pipeline (Call Flow)

```
CALLER SPEAKS
    │
    ▼
┌─────────────────────────────────────────┐
│  VOICE ACTIVITY DETECTION (VAD)         │
│  Silero VAD (open source)               │
│  Detects speech start/end               │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  STREAMING ASR (if needed)              │
│  Whisper or Cloudflare audio-to-text    │
│  Converts speech → text for agent       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  AGENT CONVERSATION ENGINE              │
│  Base44 agent processes:                │
│  1. Receives text from ASR              │
│  2. Retrieves DAIM memory for caller    │
│  3. Generates response (text)           │
│  4. Triggers voice synthesis            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  VOICE SYNTHESIS                        │
│  CSM-1B (fine-tuned on Israel's voice)  │
│  Text + speaker_id → audio codes → WAV  │
│  Streamed back in real-time             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
            CALLER HEARS RESPONSE
                  │
                  ▼
            (back to VAD for next turn)
```

### 5.3 Agent Memory Per Caller (DAIM Integration)

```
CALLER ID (phone number / user ID)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DAIM MEMORY DAG                                            │
│                                                             │
│  [Caller Profile] ──→ [Call History] ──→ [Preferences]      │
│       │                    │                    │            │
│       ▼                    ▼                    ▼            │
│  [Name, Voice     [Transcripts,     [Topics discussed,     │
│   sample, ID]      Timestamps,       Sentiment,             │
│                    Duration]          Next actions]          │
│                                                             │
│  Immutable: Every interaction is a new DAG node             │
│  Append-only: History is never overwritten                  │
│  Per-caller isolation: Different callers = different DAGs   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. BASE44 AGENT SPEC — THE VOICE AGENT CODE

### 6.1 Base44 Agent Architecture

From our recon of Base44's platform:
- **Backend-as-a-Service** — managed database, auth, functions
- **React/Next.js frontend** — exported for production hosting
- **Cloud Functions** — serverless functions for agent logic
- **REST API** — for external integrations
- **Real-time** — WebSocket support for live voice

### 6.2 Voice Agent Base Code Structure

```javascript
// ============================================================
// BASE44 VOICE AGENT — "ACORN" TEMPLATE
// File: agents/VoiceAgent.js
// Description: Base44 agent configured for voice conversation
// with CSM-1B synthesis and DAIM memory
// ============================================================

import { Base44Agent } from '@base44/sdk';
import { DAIMClient } from '@e5/daim-sdk';
import { VoiceSession } from '../voice/Session';
import { CSMVoiceEngine } from '../voice/CSMEngine';

/**
 * VoiceAgent — Base44 agent with voice capabilities
 * 
 * Inherits from Base44Agent for:
 * - Database persistence
 * - Authentication
 * - REST API endpoints
 * - Cloud function execution
 */
export class VoiceAgent extends Base44Agent {
  
  constructor(config) {
    super({
      name: config.name || 'ACORN Voice Agent',
      personality: config.personality || 'professional',
      voiceId: config.voiceId || 'israel-default',
      ...config
    });
    
    // DAIM memory client (our DAG-based memory)
    this.memory = new DAIMClient({
      endpoint: config.daimEndpoint || 'https://boardroom.iamgodiam.net/api/v1/memory',
      agentId: this.id,
    });
    
    // Voice engine (CSM-1B via Cloudflare)
    this.voiceEngine = new CSMVoiceEngine({
      modelEndpoint: config.csmEndpoint,
      speakerId: config.voiceId,
      sampleRate: 16000,
    });
    
    // Voice activity detection
    this.vad = null; // Initialized per-session
    
    // Conversation state
    this.sessions = new Map(); // callerId -> VoiceSession
  }

  // ──────────────────────────────────────────────────────────────
  // CORE VOICE METHODS
  // ──────────────────────────────────────────────────────────────

  /**
   * Initiate outbound call or handle inbound call
   * Called by: WebRTC handler, phone webhook, or API
   */
  async handleCall(callerId, audioStream) {
    // Create or retrieve session
    let session = this.sessions.get(callerId);
    if (!session) {
      session = await this.createSession(callerId);
    }
    
    // Process voice input
    const transcription = await this.voiceEngine.transcribe(audioStream);
    
    // Retrieve caller memory from DAIM DAG
    const callerMemory = await this.memory.recall(callerId, {
      limit: 10,
      contentTypes: ['episodic', 'semantic', 'identity']
    });
    
    // Build conversation context
    const context = this.buildContext(callerMemory, transcription);
    
    // Generate text response (uses Base44's built-in LLM)
    const responseText = await this.generateResponse(context);
    
    // Synthesize voice (CSM-1B fine-tuned on Israel's voice)
    const audioResponse = await this.voiceEngine.synthesize({
      text: responseText,
      speakerId: this.config.voiceId,
      // Conversation style matching
      style: this.inferStyle(context),
    });
    
    // Store interaction in DAIM DAG
    await this.memory.retain(callerId, {
      content: `Inbound: "${transcription}" → Response: "${responseText}"`,
      contentType: 'episodic',
      source: 'voice-session',
      accessTier: 1,
      metadata: {
        sessionId: session.id,
        duration: session.duration,
        sentiment: this.analyzeSentiment(transcription),
      }
    });
    
    return {
      audio: audioResponse,
      text: responseText,
      session: session.id,
    };
  }

  /**
   * Fine-grained voice control
   */
  async synthesizeWithStyle(text, options = {}) {
    return this.voiceEngine.synthesize({
      text,
      speakerId: options.speakerId || this.config.voiceId,
      emotion: options.emotion || 'neutral',     // neutral, warm, professional
      speed: options.speed || 1.0,               // 0.5 - 2.0
      pitch: options.pitch || 1.0,               // 0.5 - 2.0
      // Provide audio context for voice consistency
      audioContext: options.referenceAudio || null,
    });
  }

  /**
   * Build conversation context from DAG memory
   */
  buildContext(memory, currentInput) {
    const callerFacts = memory
      .filter(n => n.contentType === 'semantic')
      .map(n => n.content)
      .join('\n');
    
    const recentHistory = memory
      .filter(n => n.contentType === 'episodic')
      .slice(0, 5)
      .map(n => n.content)
      .join('\n');
    
    return {
      systemPrompt: `You are ${this.config.name}, ${this.config.personality}.\nCaller facts:\n${callerFacts}`,
      history: recentHistory,
      currentInput: currentInput,
      callerMood: this.analyzeSentiment(currentInput),
    };
  }

  // ──────────────────────────────────────────────────────────────
  // CALL MANAGEMENT
  // ──────────────────────────────────────────────────────────────

  async createSession(callerId) {
    const session = new VoiceSession({
      callerId,
      agentId: this.id,
      startTime: Date.now(),
    });
    this.sessions.set(callerId, session);
    return session;
  }

  async endCall(callerId) {
    const session = this.sessions.get(callerId);
    if (session) {
      // Store session summary in DAIM
      await this.memory.retain(callerId, {
        content: `Session summary: ${session.summary}`,
        contentType: 'episodic',
        source: 'session-end',
      });
      this.sessions.delete(callerId);
    }
  }

  // ──────────────────────────────────────────────────────────────
  // OUTBOUND CALLING (ACORN Outreach)
  // ──────────────────────────────────────────────────────────────

  async initiateOutreach(phoneNumber, purpose) {
    // Retrieve any existing memory for this number
    const existingMemory = await this.memory.recall(phoneNumber);
    
    // Generate opening script based on memory
    const opening = await this.generateOpening(purpose, existingMemory);
    
    // Place call via Twilio/Vonage integration
    const call = await this.dial(phoneNumber);
    
    // Send opening line
    await this.sendVoice(call.id, await this.synthesizeWithStyle(opening));
    
    return call;
  }

  // ──────────────────────────────────────────────────────────────
  // REST API ENDPOINTS (Base44 auto-generates these)
  // ──────────────────────────────────────────────────────────────

  /**
   * POST /api/v1/voice/inbound
   * Handle inbound voice call
   */
  async onVoiceInbound(req, res) {
    const { callerId, audioData } = req.body;
    const result = await this.handleCall(callerId, audioData);
    res.json(result);
  }

  /**
   * POST /api/v1/voice/outbound
   * Initiate outbound call
   */
  async onVoiceOutbound(req, res) {
    const { phoneNumber, purpose } = req.body;
    const call = await this.initiateOutreach(phoneNumber, purpose);
    res.json({ callId: call.id, status: 'initiated' });
  }

  /**
   * GET /api/v1/voice/memory/:callerId
   * Retrieve DAG memory for a caller
   */
  async onGetMemory(req, res) {
    const { callerId } = req.params;
    const memory = await this.memory.recall(callerId);
    res.json({ callerId, memories: memory });
  }

  /**
   * GET /api/v1/voice/sessions
   * List active voice sessions
   */
  async onListSessions(req, res) {
    const sessions = Array.from(this.sessions.entries()).map(([id, s]) => ({
      callerId: id,
      duration: Date.now() - s.startTime,
      turns: s.turnCount,
    }));
    res.json({ sessions, total: sessions.length });
  }
}
```

### 6.3 Cloudflare Worker — Voice API Proxy

```javascript
// ============================================================
// CLOUDFLARE WORKER — Voice API Proxy
// File: workers/voice-api.js
// Routes: *.iamgodiam.net/voice/*
// ============================================================

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Route to appropriate handler
    switch (url.pathname) {
      case '/voice/synthesize':
        return handleSynthesis(request, env);
      case '/voice/transcribe':
        return handleTranscription(request, env);
      case '/voice/vad':
        return handleVAD(request, env);
      default:
        return new Response('Not Found', { status: 404 });
    }
  }
};

/**
 * Voice Synthesis — CSM-1B via Workers AI
 * Accepts text + speaker config, returns audio
 */
async function handleSynthesis(request, env) {
  const { text, speakerId, style } = await request.json();
  
  // Option A: Use @cf/ model catalog for TTS (if CSM not available)
  // Option B: Route to dedicated CSM endpoint on Base44
  // Option C: Use R2-hosted fine-tuned model via Workers AI custom model
  
  const audioResponse = await fetch(
    `https://api.base44.com/v1/voice/synthesize`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        speaker_id: speakerId || 'israel-default',
        style: style || 'neutral',
        model: 'csm-1b',
      }),
    }
  );
  
  const audioData = await audioResponse.arrayBuffer();
  
  return new Response(audioData, {
    headers: {
      'Content-Type': 'audio/wav',
      'Cache-Control': 'public, max-age=3600',
    }
  });
}

/**
 * Voice Transcription — Whisper via Workers AI
 */
async function handleTranscription(request, env) {
  const audioData = await request.arrayBuffer();
  
  // Cloudflare Workers AI — Whisper model
  const response = await env.AI.run('@cf/openai/whisper', {
    audio: [...new Uint8Array(audioData)],
  });
  
  return Response.json({
    text: response.text,
    confidence: response.confidence || 0.95,
  });
}

/**
 * Voice Activity Detection
 */
async function handleVAD(request, env) {
  const audioChunk = await request.arrayBuffer();
  // Process VAD — detect speech/silence boundaries
  // Return timestamps for speech segments
  return Response.json({ hasSpeech: true, timestamp: Date.now() });
}
```

### 6.4 Voice Session Manager (WebRTC)

```javascript
// ============================================================
// VOICE SESSION — WebRTC Manager
// File: voice/Session.js
// ============================================================

export class VoiceSession {
  constructor({ callerId, agentId, startTime }) {
    this.id = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.callerId = callerId;
    this.agentId = agentId;
    this.startTime = startTime;
    this.turnCount = 0;
    this.transcript = [];
    this.currentState = 'listening'; // listening, processing, speaking
    
    // WebRTC peer connection
    this.peerConnection = null;
    this.audioContext = null;
  }

  get duration() {
    return Date.now() - this.startTime;
  }

  get summary() {
    return `Session ${this.id}: ${this.turnCount} turns, ${this.duration}ms`;
  }

  addTurn(inbound, outbound) {
    this.turnCount++;
    this.transcript.push({
      turn: this.turnCount,
      inbound,
      outbound,
      timestamp: Date.now(),
    });
  }

  // WebRTC setup for browser-based voice
  async initializeWebRTC() {
    this.peerConnection = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });
    
    // Handle incoming audio stream
    this.peerConnection.ontrack = (event) => {
      this.processIncomingAudio(event.streams[0]);
    };
    
    return this.peerConnection;
  }

  async processIncomingAudio(stream) {
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = this.audioContext.createMediaStreamSource(stream);
    const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    
    processor.onaudioprocess = (event) => {
      const audioData = event.inputBuffer.getChannelData(0);
      // Send to VAD → ASR → Agent pipeline
      this.onAudioData(audioData);
    };
    
    source.connect(processor);
    processor.connect(this.audioContext.destination);
  }

  onAudioData(audioData) {
    // Emit to agent for processing
    // This connects to the VoiceAgent.handleCall pipeline
  }
}
```

---

## 7. CLOUDFLARE DEPLOYMENT STRATEGY

### 7.1 Our Business Account Assets

| Asset | Status | Use |
|---|---|---|
| **Custom Domain** | ✅ `iamgodiam.net` | Voice app: `voice.iamgodiam.net` |
| **Cloudflare Tunnel** | ✅ `boardroom.iamgodiam.net` | Internal API routing |
| **Workers AI** | ✅ Available | Whisper transcription, TTS |
| **Pages** | ✅ Available | Host voice app frontend |
| **R2** | ✅ Available | Store voice recordings |
| **D1** | ✅ Available | Call metadata, session records |
| **KV** | ✅ Available | Session state, ephemeral data |
| **Vectorize** | ✅ Available | Voice memory embeddings |
| **AI Gateway** | ✅ Available | Cache, rate-limit, log AI calls |

### 7.2 Deployment Map

```
voice.iamgodiam.net          → Cloudflare Pages (React/Next.js frontend)
voice.iamgodiam.net/api/*    → Cloudflare Workers (API layer)
voice.iamgodiam.net/ws/*     → Cloudflare Durable Objects (WebRTC signaling)
csm.iamgodiam.net           → Base44 Cloud Function (CSM-1B inference)
daim.iamgodiam.net          → Cloudflare Tunnel → Boardroom (DAG memory)
recordings.iamgodiam.net    → R2 (voice recording storage)
```

### 7.3 Workers AI Model Options for Voice

Available NOW on Workers AI:

| Model | Type | Use | Cost |
|---|---|---|---|
| `@cf/openai/whisper` | ASR | Speech → Text | Free tier available |
| `@cf/openai/whisper-tiny-en` | ASR | Fast English ASR | Free tier |
| `@cf/microsoft/resnet-50` | CV | Not voice-relevant | - |
| `@cf/meta/llama-3.1-8b-instruct` | LLM | Agent reasoning | $0.0001/1K tokens |
| `@cf/meta/llama-3.2-3b-instruct` | LLM | Lightweight agent | Cheaper |
| Custom model (fine-tuned) | Custom | CSM-1B (if uploaded) | Depends on size |

**For CSM-1B specifically:** Workers AI's model catalog may not include CSM-1B natively. Options:
1. **Cloudflare Custom Model** — Upload fine-tuned CSM-1B as a custom model (needs approval)
2. **External inference** — Host CSM-1B on Base44 GPU workers, proxy via Cloudflare
3. **Hybrid** — Use Workers AI for ASR + LLM, external API for voice synthesis

---

## 8. COMPUTE REQUIREMENTS & HARDWARE PLAN

### 8.1 Fine-Tuning CSM-1B on Israel's Voice

| Phase | Task | GPU | VRAM | Time | Cost |
|---|---|---|---|---|---|
| **Data prep** | Record + clean 30min-2hr of Israel speaking | CPU | - | 1-2 days | Free |
| **Tokenization** | Mimi encode audio → RVQ codes | CPU/GPU | 8GB | 2-4 hrs | ~$5 |
| **Fine-tuning** | CSM-1B LoRA fine-tuning | A100/V100 | 40-80GB | 6-24 hrs | ~$20-50 |
| **Evaluation** | Generate test samples, measure quality | GPU | 16GB | 1-2 hrs | ~$5 |
| **Iteration** | Adjust hyperparameters, re-train | GPU | 40-80GB | 6-12 hrs | ~$20-40 |

**Total estimated cost: $50-100 per voice model**

### 8.2 Production Inference (Real-Time Voice)

| Component | Compute | Latency Target | Cost/Month |
|---|---|---|---|
| **ASR (Whisper)** | Workers AI | < 200ms | ~$50 |
| **Agent (Llama 8B)** | Workers AI | < 500ms | ~$100 |
| **Voice (CSM-1B)** | GPU instance | < 300ms | ~$200-500 |
| **Memory (DAIM)** | PostgreSQL | < 50ms | ~$20 |

**Total monthly cost: ~$370-670 for moderate usage (1000 calls/day)**

### 8.3 GPU Options

| Provider | GPU | Cost/hr | Notes |
|---|---|---|---|
| **Cloudflare Workers AI** | Shared | Pay-per-request | No management, but may not support CSM |
| **Modal** | A100 80GB | $2.50 | Good for fine-tuning |
| **Lambda Cloud** | A100 80GB | $1.10 | Cheapest bare-metal GPU |
| **RunPod** | A100 80GB | $0.79-2.50 | Community GPUs, variable |
| **Vast.ai** | A100 80GB | $0.50-1.50 | Cheapest, less reliable |
| **Base44 compute** | Varies | Included | Check if GPU workers available |

---

## 9. VOICE DATA PIPELINE — FINE-TUNING ISRAEL'S VOICE

### 9.1 Data Collection Strategy

| Source | Quality | Quantity | Notes |
|---|---|---|---|
| **WhatsApp voice notes** | High | Variable | Already exist, Israel's natural speaking style |
| **Meeting recordings** | High | Variable | Board calls, strategy sessions |
| **Phone calls** | High | Variable | Record with consent |
| **Scripted reading** | Highest | 30min-1hr | Read diverse text for coverage |
| **Conversational pairs** | High | 1-2 hrs | Natural dialogue for CSM training |

### 9.2 Data Processing Pipeline

```
RAW AUDIO RECORDINGS
    │
    ▼
┌─────────────────────────────────────────┐
│  1. AUDIO CLEANING                      │
│  - Remove background noise              │
│  - Normalize volume                     │
│  - Detect and remove silence            │
│  - Split into segments (5-15 sec)      │
│  - Quality filter (SNR > 20dB)         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  2. TRANSCRIPTION                       │
│  - Whisper large-v3 for ASR            │
│  - Manual correction for accuracy       │
│  - Align text to audio timestamps       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  3. METADATA CREATION                   │
│  - Speaker ID: [israel]                │
│  - Emotion/style tags                   │
│  - Topic classification                 │
│  - Quality score                        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  4. MIMI TOKENIZATION                   │
│  - Encode audio → RVQ codes (32 CB)    │
│  - Store as num_codebooks × seq_len    │
│  - Pack with Llama-3 text tokens       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  5. CSM-1B FINE-TUNING                  │
│  - Start from pretrained CSM-1B        │
│  - LoRA on backbone attention layers    │
│  - Train for 25-50 epochs               │
│  - Hyperparameter sweep (Optuna)        │
│  - Generate test samples each epoch     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  6. EVALUATION                          │
│  - MOS (Mean Opinion Score) for quality │
│  - Speaker similarity (embeddings)      │
│  - Latency benchmark                    │
│  - Naturalness rating                   │
└─────────────────────────────────────────┘
```

---

## 10. FULL IMPLEMENTATION ROADMAP

### Sprint 0: Data Collection (Week 1)
| Task | Owner | Deliverable |
|---|---|---|
| Record Israel's voice (scripted reading) | Israel | 1hr clean audio |
| Collect existing voice notes/recordings | Sue | Raw audio dataset |
| Clean and segment audio | Forge | Processed audio files |
| Transcribe and align | Forge | Text-audio pairs |

### Sprint 1: Model Fine-Tuning (Week 2)
| Task | Owner | Deliverable |
|---|---|---|
| Set up GPU environment (Modal/Lambda) | Draco | Running GPU instance |
| Tokenize audio with Mimi | Forge | Tokenized dataset |
| Fine-tune CSM-1B (initial) | Forge | Fine-tuned checkpoint v0 |
| Generate test samples | Forge | Sample audio files |
| Evaluate and iterate | Scout | Quality report |

### Sprint 2: Agent Integration (Week 3)
| Task | Owner | Deliverable |
|---|---|---|
| Create VoiceAgent Base44 class | Sue | Agent code |
| Connect to CSM-1B inference endpoint | Forge | Voice pipeline |
| Integrate DAIM memory | Hermie | Memory-augmented agent |
| WebRTC frontend | Miranda | Browser voice UI |
| REST API endpoints | Sue | Full API |

### Sprint 3: Deployment (Week 4)
| Task | Owner | Deliverable |
|---|---|---|
| Deploy to Cloudflare Pages | Miranda | Live frontend |
| Configure Workers AI | Draco | Production inference |
| Set up custom domain | Draco | `voice.iamgodiam.net` |
| SSL + security headers | Draco | Production hardening |
| Monitoring (Sentry + Analytics) | Draco | Observability |

### Sprint 4: ACORN Outreach (Week 5-6)
| Task | Owner | Deliverable |
|---|---|---|
| Build outbound call system | Sue | Call center feature |
| Integrate phone numbers (Twilio) | Sue | Phone connectivity |
| ACORN outreach campaign builder | Miranda | Campaign UI |
| A/B testing framework | Scout | Test infrastructure |
| Analytics dashboard | Miranda | Metrics UI |

---

## 11. RISK REGISTER & UNK UNKS

| Risk | Impact | Probability | Mitigation |
|---|---|---|---|
| CSM-1B quality insufficient | High | Medium | Fine-tune longer, consider larger model |
| Cloudflare Workers AI latency too high | Medium | Medium | Use external GPU, cache aggressively |
| Voice data insufficient quantity | High | Low | Record more, use data augmentation |
| Fine-tuning compute costs exceed budget | Medium | Medium | Use Vast.ai spot instances |
| Sesame license changes (Apache 2.0 now) | Low | Low | Code is open source, fork if needed |
| WebRTC complexity | Medium | Low | Use library (simple-peer, twilio-video) |
| Call quality over phone networks | Medium | High | Use Opus codec, adaptive bitrate |
| Regulatory (TCPA for outbound calls) | High | Medium | Get consent, honor DNC list |
| Voice cloning ethics/legal | Medium | Medium | Only use Israel's voice with consent |

---

## 12. SUCCESS CRITERIA & KPIs

| Metric | Target | Measurement |
|---|---|---|
| **Voice MOS** | ≥ 4.0/5.0 | Mean Opinion Score test |
| **Speaker similarity** | ≥ 85% | Embedding cosine similarity |
| **End-to-end latency** | < 1 second | Audio in → audio out |
| **ASR accuracy** | ≥ 95% | WER on test set |
| **Agent coherence** | ≥ 4.0/5.0 | Human evaluation |
| **Call completion rate** | ≥ 80% | ACORN outreach metric |
| **Cost per call** | < $0.10 | Infrastructure cost |
| **Uptime** | ≥ 99.5% | Monitoring |

---

## 13. SITUATION ROOM — TEAM ASSIGNMENTS

| Role | Agent | Sprint 0-1 | Sprint 2-3 | Sprint 4 |
|---|---|---|---|---|
| **Visionary / Think Tank** | Hermie | Architecture review | Integration oversight | Strategy |
| **CEO / Executor** | Sue | Data collection | Agent deployment | ACORN campaigns |
| **ML Engineer** | Forge | Fine-tuning | Voice integration | Optimization |
| **Researcher** | Scout | Model eval | Benchmarking | A/B testing |
| **Content / UI** | Miranda | - | WebRTC frontend | Dashboard |
| **DevOps** | Draco | GPU setup | Cloudflare deploy | Monitoring |

---

## APPENDIX A: BASE44 AGENT CONFIGURATION

```yaml
# base44.config.yaml
# Base44 Agent Configuration for ACORN Voice Platform

app:
  name: "ACORN Voice Platform"
  subdomain: "acorn-voice"
  
database:
  type: "postgresql"
  tables:
    - name: "callers"
      columns:
        - { name: "id", type: "uuid", primary: true }
        - { name: "phone", type: "string", unique: true }
        - { name: "name", type: "string" }
        - { name: "daim_agent_id", type: "string" }
        
    - name: "calls"
      columns:
        - { name: "id", type: "uuid", primary: true }
        - { name: "caller_id", type: "uuid", foreign: "callers.id" }
        - { name: "direction", type: "enum", values: ["inbound", "outbound"] }
        - { name: "duration_ms", type: "integer" }
        - { name: "transcript", type: "jsonb" }
        - { name: "audio_url", type: "string" } # R2 URL
        - { name: "outcome", type: "string" }
        
    - name: "outreach_campaigns"
      columns:
        - { name: "id", type: "uuid", primary: true }
        - { name: "name", type: "string" }
        - { name: "script", type: "text" }
        - { name: "voice_id", type: "string" }
        - { name: "status", type: "enum", values: ["draft", "active", "paused"] }
        
functions:
  - name: "synthesize_voice"
    runtime: "python3.12"
    handler: "voice.synthesize"
    timeout: 30
    
  - name: "process_call"
    runtime: "python3.12"
    handler: "agent.process_call"
    timeout: 300
    
  - name: "initiate_outbound"
    runtime: "python3.12"
    handler: "agent.outbound"
    timeout: 60

environment:
  CSM_ENDPOINT: "https://csm.iamgodiam.net"
  DAIM_ENDPOINT: "https://daim.iamgodiam.net"
  WHISPER_MODEL: "@cf/openai/whisper"
  AGENT_MODEL: "@cf/meta/llama-3.1-8b-instruct"
  VOICE_ID: "israel-default"
  R2_BUCKET: "acorn-recordings"
  D1_DATABASE: "acorn-db"
```

---

*Document committed to: github.com/IAMGODIAM/voice-agent-platform*  
*Wyrmcore Protocol v1.0 | Full Chairman Protocol Authorized*  
*Visionary: Hermie | Executor: Sue*
