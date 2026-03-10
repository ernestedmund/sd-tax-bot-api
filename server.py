"""
SD County Property Tax Assistant - FastAPI Backend
===================================================
Wraps rag_engine.py as a REST API.

Endpoints:
  GET  /health        — health check
  POST /chat          — web chatbot (GitHub Pages frontend)
  POST /voice         — legacy Twilio voice (Say/Gather, kept as fallback)
  POST /gather        — legacy Twilio gather handler
  POST /voice_relay   — ConversationRelay entry point (new primary voice bot)
  WS   /ws            — ConversationRelay websocket handler (ElevenLabs TTS)

Run locally:
  pip install fastapi uvicorn anthropic twilio websockets python-multipart
  ANTHROPIC_API_KEY=sk-... uvicorn server:app --reload
"""

import os
import json
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from anthropic import Anthropic
from twilio.twiml.voice_response import VoiceResponse, Gather, Connect

# Import RAG components from rag_engine.py (same directory)
from rag_engine import (
    parse_knowledge_base,
    build_tfidf_index,
    retrieve,
    build_system_prompt,
    RAW_KNOWLEDGE_BASE,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="County Property Tax Assistant")

ALLOWED_ORIGINS = [
    "https://ernestedmund.github.io",
    "https://propertytaxfaq.com",
    "https://www.propertytaxfaq.com",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

# Build index once at startup (not per-request)
KB_CHUNKS = parse_knowledge_base(RAW_KNOWLEDGE_BASE)
KB_INDEX = build_tfidf_index(KB_CHUNKS)

print(f"Loaded {len(KB_CHUNKS)} knowledge base chunks at startup.")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT_WINDOW = 3600
RATE_LIMIT_MAX    = 15

ip_request_log: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    ip_request_log[ip] = [t for t in ip_request_log[ip] if t > window_start]
    if len(ip_request_log[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached. Please try again later or contact your county assessor's office."
        )
    ip_request_log[ip].append(now)


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

sessions: dict[str, list[dict]] = {}
MAX_HISTORY_TURNS = 6


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    chunks_used: list[str]


# ---------------------------------------------------------------------------
# Shared RAG + Claude helpers
# ---------------------------------------------------------------------------

VOICE_SYSTEM_ADDENDUM = """
You are answering a phone call, so follow these rules strictly:
- Respond in plain spoken English only. No bullet points, numbered lists, headers, or markdown.
- Keep your answer to 2 to 3 sentences. Be concise -- the caller is listening, not reading.
- Never read out source citations, rule numbers, form names, or publication references.
- If a phone number is needed, speak it naturally: "call six one nine, two three six, three seven seven one".
- Never end with a standalone closing sentence. Instead, trail your final answer sentence into an invitation for more questions using a comma and a short tag like "...or let me know if you have another question" or "...or feel free to ask anything else."
"""

EXPANSION_PROMPT = """You are a query rewriter for a property tax assistant.
Given a conversation history and a short follow-up message, rewrite the follow-up as a single clear, standalone question that captures the user's full intent.
Output ONLY the rewritten question -- no explanation, no preamble."""

def expand_query(message: str, history: list[dict]) -> str:
    if len(message.split()) > 10 or not history:
        return message

    recent = history[-4:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=60,
        system=EXPANSION_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Conversation so far:\n{history_text}\n\nFollow-up message: {message}"
        }]
    )
    expanded = response.content[0].text.strip()
    print(f"[query expansion] '{message}' -> '{expanded}'")
    return expanded if expanded else message


def get_rag_reply(message: str, history: list[dict], voice: bool = False) -> tuple[str, list[str]]:
    retrieval_query = expand_query(message, history)
    retrieved = retrieve(retrieval_query, KB_CHUNKS, KB_INDEX, top_k=4)
    system_prompt = build_system_prompt(retrieved)
    if voice:
        system_prompt = system_prompt + VOICE_SYSTEM_ADDENDUM
    chunk_ids = [c["id"] for c in retrieved]

    trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]
    trimmed_history.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300 if voice else 600,
        system=system_prompt,
        messages=trimmed_history,
    )
    return response.content[0].text, chunk_ids


# ---------------------------------------------------------------------------
# Endpoints -- web chat
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(KB_CHUNKS)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host
    check_rate_limit(client_ip)

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 characters).")

    session_id = req.session_id[:64]
    history = sessions.get(session_id, [])

    reply, chunk_ids = get_rag_reply(message, history)

    full_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    sessions[session_id] = full_history[-(MAX_HISTORY_TURNS * 2):]

    return ChatResponse(reply=reply, chunks_used=chunk_ids)


# ---------------------------------------------------------------------------
# Endpoints -- legacy Twilio voice (Say/Gather fallback)
# ---------------------------------------------------------------------------

LEGACY_VOICE = "Google.en-US-Chirp3-HD-Leda"
GATHER_TIMEOUT = 5
GATHER_SPEECH_TIMEOUT = "auto"
GREETING_LEGACY = (
    "Hi, you've reached the County Property Tax Assistant. "
    "What's your property tax question?"
)
NO_INPUT = "I didn't catch that. Go ahead and ask your question."
TRANSFER_NUMBER = ""  # Set to your county assessor's number


def twiml_response(twiml: VoiceResponse) -> Response:
    return Response(content=str(twiml), media_type="application/xml")


@app.post("/voice")
async def voice_entry(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    sessions.pop(call_sid, None)

    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/gather",
        method="POST",
        timeout=GATHER_TIMEOUT,
        speech_timeout=GATHER_SPEECH_TIMEOUT,
        language="en-US",
    )
    gather.say(GREETING_LEGACY, voice=LEGACY_VOICE)
    vr.append(gather)
    vr.redirect("/voice", method="POST")
    return twiml_response(vr)


@app.post("/gather")
async def gather_handler(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    speech_result = (form.get("SpeechResult") or "").strip()
    digits = (form.get("Digits") or "").strip()

    vr = VoiceResponse()

    if digits == "0":
        vr.say("Transferring you now. Please hold.", voice=LEGACY_VOICE)
        vr.dial(TRANSFER_NUMBER)
        return twiml_response(vr)

    if not speech_result:
        g = Gather(
            input="speech dtmf",
            action="/gather",
            method="POST",
            timeout=GATHER_TIMEOUT,
            speech_timeout=GATHER_SPEECH_TIMEOUT,
            language="en-US",
        )
        g.say(NO_INPUT, voice=LEGACY_VOICE)
        vr.append(g)
        vr.redirect("/voice", method="POST")
        return twiml_response(vr)

    history = sessions.get(call_sid, [])
    try:
        reply, chunk_ids = get_rag_reply(speech_result, history, voice=True)
        print(f"[legacy {call_sid}] Q: {speech_result[:80]}")
        print(f"[legacy {call_sid}] Chunks: {chunk_ids}")
        print(f"[legacy {call_sid}] A: {reply[:120]}")
    except Exception as e:
        print(f"[legacy {call_sid}] Error: {e}")
        reply = (
            "I'm sorry, I had trouble looking that up. "
            "Please contact your county assessor's office directly for assistance."
        )

    updated_history = history + [
        {"role": "user", "content": speech_result},
        {"role": "assistant", "content": reply},
    ]
    sessions[call_sid] = updated_history[-(MAX_HISTORY_TURNS * 2):]

    g = Gather(
        input="speech dtmf",
        action="/gather",
        method="POST",
        timeout=GATHER_TIMEOUT,
        speech_timeout=GATHER_SPEECH_TIMEOUT,
        language="en-US",
    )
    g.say(reply, voice=LEGACY_VOICE)
    g.say("Any other questions?", voice=LEGACY_VOICE)
    vr.append(g)
    vr.redirect("/voice", method="POST")
    return twiml_response(vr)


# ---------------------------------------------------------------------------
# Endpoints -- ConversationRelay (ElevenLabs TTS via websocket)
# ---------------------------------------------------------------------------

ELEVENLABS_VOICE_ID = "EST9Ui6982FZPSi7gCHi"
VOICE_SETTINGS = "0.9_0.75_0.75"   # speed_stability_similarity
ELEVENLABS_VOICE = f"{ELEVENLABS_VOICE_ID}-{VOICE_SETTINGS}"

GREETING_RELAY = (
    "Hi, you've reached the County Property Tax Assistant. "
    "What's your property tax question?"
)


@app.post("/voice_relay")
async def voice_relay(request: Request):
    """
    ConversationRelay entry point. Returns TwiML that hands the call off
    to a websocket, which drives the live conversation with ElevenLabs TTS.
    """
    print("[voice_relay] Incoming call received")
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    print(f"[voice_relay] CallSid={call_sid}")
    sessions.pop(call_sid, None)

    # Railway automatically sets RAILWAY_PUBLIC_DOMAIN — use it for the websocket URL
    # so Twilio gets the correct public hostname, not an internal Railway address
    host = os.environ.get("RAILWAY_PUBLIC_DOMAIN") or request.headers.get("host", "")
    ws_url = f"wss://{host}/ws"
    print(f"[relay] WebSocket URL: {ws_url}")

    vr = VoiceResponse()
    connect = Connect()
    connect.conversation_relay(
        url=ws_url,
        tts_provider="ElevenLabs",
        voice=ELEVENLABS_VOICE,
        language="en-US",
        transcription_provider="deepgram",
        speech_model="nova-2",
        welcome_greeting=GREETING_RELAY,
    )
    vr.append(connect)
    return twiml_response(vr)


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """
    ConversationRelay websocket handler.

    Twilio sends JSON messages:
      setup    -- call metadata, extract CallSid
      prompt   -- transcribed caller speech, run RAG + Claude, reply with text
      interrupt-- caller interrupted, log only
      dtmf     -- keypad digit

    We reply with JSON:
      {"type": "text", "token": "<reply text>"}   -- Twilio speaks via ElevenLabs
      {"type": "end"}                              -- hang up / end relay
    """
    await websocket.accept()
    call_sid = "unknown"

    try:
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = msg.get("type") or msg.get("event", "")

            # Debug: log every incoming message so we can see exact field names
            print(f"[relay debug] event='{event}' keys={list(msg.keys())} raw={json.dumps(msg)[:400]}")

            if event == "setup":
                call_sid = msg.get("callSid") or msg.get("CallSid", "unknown")
                print(f"[relay] Connected: {call_sid}")

            elif event == "prompt":
                utterance = (msg.get("voicePrompt") or msg.get("speech", "") or msg.get("text", "")).strip()
                if not utterance:
                    continue

                print(f"[relay {call_sid}] Q: {utterance[:80]}")
                history = sessions.get(call_sid, [])

                try:
                    reply, chunk_ids = get_rag_reply(utterance, history, voice=True)
                    print(f"[relay {call_sid}] Chunks: {chunk_ids}")
                    print(f"[relay {call_sid}] A: {reply[:120]}")
                except Exception as e:
                    print(f"[relay {call_sid}] RAG error: {e}")
                    reply = (
                        "I'm sorry, I had trouble looking that up. "
                        "Please contact your county assessor's office directly for assistance."
                    )

                sessions[call_sid] = (history + [
                    {"role": "user", "content": utterance},
                    {"role": "assistant", "content": reply},
                ])[-(MAX_HISTORY_TURNS * 2):]

                await websocket.send_text(json.dumps({"type": "text", "token": reply, "last": True}))

            elif event == "dtmf":
                digit = msg.get("digit", "")
                if digit == "0":
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": "Transferring you now. Please hold.",
                        "last": True
                    }))
                    await websocket.send_text(json.dumps({"type": "end"}))

            elif event == "interrupt":
                print(f"[relay {call_sid}] Interrupted")

    except WebSocketDisconnect:
        print(f"[relay {call_sid}] Disconnected")
    except Exception as e:
        print(f"[relay {call_sid}] Error: {e}")
