"""
SD County Property Tax Assistant - FastAPI Backend
===================================================
Wraps rag_engine.py as a REST API.
The GitHub Pages frontend calls POST /chat with the user message;
this server does retrieval and calls Claude Haiku, returning only the reply.

Your Anthropic API key stays server-side — never exposed to the browser.

Twilio voice routes added:
  POST /voice  — entry point for inbound calls (greet + gather speech)
  POST /gather — receives transcribed speech, runs RAG, speaks answer, loops

Deploy options for SD County:
  - Azure App Service (likely already in your IT contract)
  - AWS Lambda + API Gateway (very cheap for this traffic volume)
  - Any Linux VM or container behind county firewall

Run locally:
  pip install fastapi uvicorn anthropic twilio
  ANTHROPIC_API_KEY=sk-... uvicorn server:app --reload
"""

import os
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from anthropic import Anthropic
from twilio.twiml.voice_response import VoiceResponse, Gather

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

app = FastAPI(title="SD County Property Tax Assistant")

ALLOWED_ORIGINS = [
    "https://ernestedmund.github.io",
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
# Rate limiting — simple in-memory (swap for Redis in production)
# Limits: 15 questions per IP per hour
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
            detail="Rate limit reached. Please try again later or call (619) 236-3771."
        )
    ip_request_log[ip].append(now)


# ---------------------------------------------------------------------------
# Session store — shared by both web chat and phone sessions
# ---------------------------------------------------------------------------

sessions: dict[str, list[dict]] = {}
MAX_HISTORY_TURNS = 6


# ---------------------------------------------------------------------------
# Request / Response models (web chat)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    chunks_used: list[str]


# ---------------------------------------------------------------------------
# Shared RAG + Claude helper (used by both /chat and /gather)
# ---------------------------------------------------------------------------

def get_rag_reply(message: str, history: list[dict]) -> tuple[str, list[str]]:
    """Run RAG retrieval and call Claude. Returns (reply, chunk_ids)."""
    retrieved = retrieve(message, KB_CHUNKS, KB_INDEX, top_k=4)
    system_prompt = build_system_prompt(retrieved)
    chunk_ids = [c["id"] for c in retrieved]

    trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]
    trimmed_history.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_prompt,
        messages=trimmed_history,
    )
    return response.content[0].text, chunk_ids


# ---------------------------------------------------------------------------
# Endpoints — web chat
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
# Endpoints — Twilio voice
# ---------------------------------------------------------------------------

VOICE = "Polly.Joanna"          # AWS Polly via Twilio — clear, neutral US English
GATHER_TIMEOUT = 5              # seconds of silence before Twilio stops listening
GATHER_SPEECH_TIMEOUT = "auto"  # Twilio auto-detects end of speech

GREETING = (
    "Hello, and thank you for calling the San Diego County Property Tax Assistant. "
    "You can ask me questions about property tax bills, exemptions, Proposition 13, "
    "assessed values, or payment deadlines. "
    "Please ask your question after the tone."
)

NO_INPUT = (
    "I didn't catch that. Please ask your question after the tone, "
    "or press zero to be transferred to the Assessor's office."
)

TRANSFER_NUMBER = "+16192363771"  # SD County Assessor


def twiml_response(twiml: VoiceResponse) -> Response:
    return Response(content=str(twiml), media_type="application/xml")


@app.post("/voice")
async def voice_entry(request: Request):
    """
    Twilio calls this when a call comes in.
    Greet the caller and open a speech gather.
    CallSid is used as the session key so conversation history persists
    across turns within the same call.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    # Clear any old session for this CallSid (fresh call)
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
    gather.say(GREETING, voice=VOICE)
    vr.append(gather)

    # If caller says nothing at all, loop back
    vr.redirect("/voice", method="POST")

    return twiml_response(vr)


@app.post("/gather")
async def gather(request: Request):
    """
    Twilio posts here with SpeechResult (transcribed caller speech).
    Run RAG + Claude, speak the answer, then loop back for another question.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    speech_result = (form.get("SpeechResult") or "").strip()
    digits = (form.get("Digits") or "").strip()

    vr = VoiceResponse()

    # Caller pressed 0 — transfer to Assessor's office
    if digits == "0":
        vr.say("Transferring you now. Please hold.", voice=VOICE)
        vr.dial(TRANSFER_NUMBER)
        return twiml_response(vr)

    # No speech captured
    if not speech_result:
        gather = Gather(
            input="speech dtmf",
            action="/gather",
            method="POST",
            timeout=GATHER_TIMEOUT,
            speech_timeout=GATHER_SPEECH_TIMEOUT,
            language="en-US",
        )
        gather.say(NO_INPUT, voice=VOICE)
        vr.append(gather)
        vr.redirect("/voice", method="POST")
        return twiml_response(vr)

    # Run RAG + Claude
    history = sessions.get(call_sid, [])
    try:
        reply, chunk_ids = get_rag_reply(speech_result, history)
        print(f"[{call_sid}] Q: {speech_result[:80]}")
        print(f"[{call_sid}] Chunks: {chunk_ids}")
        print(f"[{call_sid}] A: {reply[:120]}")
    except Exception as e:
        print(f"[{call_sid}] Error: {e}")
        reply = (
            "I'm sorry, I had trouble looking that up. "
            "Please call the Assessor's office directly at 619-236-3771."
        )

    # Update session history
    updated_history = history + [
        {"role": "user", "content": speech_result},
        {"role": "assistant", "content": reply},
    ]
    sessions[call_sid] = updated_history[-(MAX_HISTORY_TURNS * 2):]

    # Speak the answer, then gather the next question
    gather = Gather(
        input="speech dtmf",
        action="/gather",
        method="POST",
        timeout=GATHER_TIMEOUT,
        speech_timeout=GATHER_SPEECH_TIMEOUT,
        language="en-US",
    )
    gather.say(reply, voice=VOICE)
    gather.say(
        "Do you have another question? Please ask after the tone, "
        "or press zero to speak with someone at the Assessor's office.",
        voice=VOICE,
    )
    vr.append(gather)
    vr.redirect("/voice", method="POST")

    return twiml_response(vr)
