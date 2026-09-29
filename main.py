"""
Urdu Medical Scribe - backend

Receives a short audio clip (one speaking turn) from the browser, translates the
spoken Urdu into English with faster-whisper, optionally cleans up medical terms
with a local LLM (Ollama), and returns the English text plus any medical terms found.
Also generates a structured clinical note from the full transcript via Ollama.

RUN: OLLAMA_MODEL=qwen2.5:7b WHISPER_MODEL=large-v3 WHISPER_DEVICE=cpu WHISPER_COMPUTE=int8 uvicorn main:app --port 8000
"""

import io
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scribe")

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------- configuration
# tiny | base | small | medium | large-v3   (bigger = more accurate, slower)
# NOTE: distil-* models cannot translate, so don't use them here.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto | cpu | cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")  # auto | int8 | float16 ...

# Optional second pass that fixes medical wording, and powers the clinical note.
# Example:  OLLAMA_MODEL=qwen2.5:7b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# How many terms from terms.txt are fed to Whisper as a hint (its prompt window is small).
PROMPT_TERM_COUNT = int(os.getenv("PROMPT_TERM_COUNT", "60"))

MIN_AUDIO_BYTES = 2000
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# ---------------------------------------------------------------- medical lexicon


def load_terms() -> list[str]:
    path = BASE_DIR / "terms.txt"
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


TERMS = load_terms()
PROMPT_TERMS = TERMS[:PROMPT_TERM_COUNT]

# one compiled pattern per term, matched as whole words, case-insensitive
TERM_PATTERNS = [
    (t, re.compile(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", re.IGNORECASE))
    for t in TERMS
]

WHISPER_PROMPT = (
    "Medical consultation between a doctor and a patient. "
    + (", ".join(PROMPT_TERMS) + "." if PROMPT_TERMS else "")
)


def find_terms(text: str) -> list[str]:
    found = []
    for term, pattern in TERM_PATTERNS:
        if pattern.search(text):
            found.append(term)
    return found


# ---------------------------------------------------------------- model lifecycle
model: WhisperModel | None = None
model_lock = threading.Lock()  # one transcription at a time


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    log.info("Loading Whisper model '%s' (first run downloads it)...", WHISPER_MODEL)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    log.info("Model ready. Loaded %d medical terms.", len(TERMS))
    yield


app = FastAPI(title="Urdu Medical Scribe", lifespan=lifespan)

# ---------------------------------------------------------------- optional LLM cleanup
LLM_SYSTEM_PROMPT = (
    "You edit English machine translations of medical consultations that were spoken in Urdu. "
    "Fix medical terminology, drug names and obvious translation errors "
    "(for example 'sugar disease' -> 'diabetes' when that is clearly meant). "
    "Never add, remove or invent information. Never change numbers, doses or durations. "
    "Return only the corrected text, with no commentary."
)


def get_llm_prompt() -> str:
    prompt = LLM_SYSTEM_PROMPT
    if TERMS:
        prompt += (
            "\n\nKnown local medical terms and drugs (prioritize these if they sound "
            "similar to mistranslated words):\n" + ", ".join(TERMS)
        )
    return prompt


def refine_with_llm(text: str) -> tuple[str, bool]:
    if not OLLAMA_MODEL or not text:
        return text, False
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": get_llm_prompt()},
                    {"role": "user", "content": text},
                ],
            },
            timeout=60,
        )
        r.raise_for_status()
        out = r.json()["message"]["content"].strip()
        return (out, True) if out else (text, False)
    except Exception as exc:  # LLM is optional; never fail the request because of it
        log.warning("Ollama refinement skipped: %s", exc)
        return text, False


# ---------------------------------------------------------------- clinical note (summarize)
class SummarizeRequest(BaseModel):
    transcript: str


SUMMARY_SYSTEM_PROMPT = (
    "You are an expert medical scribe. Given the raw, unlabeled transcript of a consultation between a doctor and a patient, "
    "generate a structured clinical note. (You must infer who is speaking based on the context).\n\n"
    "Important requirements:\n"
    "- Do not blindly transcribe every sentence or include conversational fillers.\n"
    "- Preserve clinically important information exactly, especially medication names, doses, units, and durations.\n"
    "- Do not invent symptoms, diagnoses, or medications. Do not infer a diagnosis unless explicitly stated by the doctor.\n"
    "- Combine information from multiple turns and avoid unnecessary repetition.\n"
    "- Use concise, professional medical language. Organize into appropriate sections with bold headers (e.g. **Chief Complaint**, **History of Present Illness**, **Examination / Vitals**, **Investigations**, **Treatment / Medication Plan**, **Follow-up Plan**).\n"
    "- Only include sections that have relevant information.\n\n"
    "Return the result as a JSON object with exactly two string keys:\n"
    "1. 'chief_complaint': containing patient-reported information like Chief Complaint, History of Present Illness, Symptoms, Past Medical History, etc.\n"
    "2. 'examination_findings': containing doctor-provided information like Assessment, Examination Findings, Investigations, Diagnosis, Treatment Plan, and Follow-up. (If the doctor has not spoken yet, leave this completely empty).\n"
    "Do not include any other text."
)


@app.post("/api/summarize")
def summarize_consultation(req: SummarizeRequest):
    if not OLLAMA_MODEL:
        raise HTTPException(503, "Ollama is not configured. Set OLLAMA_MODEL to enable the clinical note.")

    if not req.transcript.strip():
        raise HTTPException(400, "There's no transcript yet to summarize.")

    transcript = req.transcript

    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "options": {"temperature": 0},
                "format": "json",
                "messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()["message"]["content"].strip()
        parsed = json.loads(out)
    except json.JSONDecodeError as exc:
        log.exception("Summarizer returned non-JSON output")
        raise HTTPException(500, f"The note generator returned an unexpected format: {exc}")
    except Exception as exc:
        log.exception("Summarization failed")
        raise HTTPException(500, f"Could not summarize: {exc}")

    return {
        "chief_complaint": parsed.get("chief_complaint", ""),
        "examination_findings": parsed.get("examination_findings", ""),
    }


# ---------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "whisper_model": WHISPER_MODEL,
        "llm_cleanup": OLLAMA_MODEL or None,
        "note_generation": bool(OLLAMA_MODEL),
        "terms_loaded": len(TERMS),
    }


@app.post("/api/transcribe")
def transcribe(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "The speech model is still loading. Try again in a moment.")

    data = audio.file.read()
    if len(data) < MIN_AUDIO_BYTES:
        raise HTTPException(400, "The recording is too short or empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "The recording is too large.")

    try:
        with model_lock:
            segments, info = model.transcribe(
                io.BytesIO(data),
                task="translate",  # Urdu speech -> English text
                language="ur",
                initial_prompt=WHISPER_PROMPT,
                beam_size=5,
                vad_filter=True,  # skips silence, which prevents made-up text
                condition_on_previous_text=False,
            )
            raw = " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        log.exception("Transcription failed")
        raise HTTPException(500, f"Could not process the audio: {exc}")

    text, refined = refine_with_llm(raw)

    return {
        "text": text,
        "refined": refined,
        "terms": find_terms(text),
        "duration": round(info.duration, 1),
    }


# Serve the front end from the same origin (avoids CORS, and microphone access
# works on http://localhost).
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")