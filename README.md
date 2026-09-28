# Urdu Medical Scribe

Doctor and patient speak Urdu. The app records each turn, translates it to English with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and puts the text in the
correct box. Medical terms are detected and listed under the boxes.

```
urdu-medical-scribe/
├── main.py            FastAPI backend (Whisper + optional Ollama cleanup)
├── terms.txt          Medical vocabulary (edit freely)
├── requirements.txt
└── static/
    └── index.html     The whole front end (HTML + CSS + JS)
```

## 1. Setup (one time)

Requires Python 3.9+.

```bash
cd urdu-medical-scribe
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

No separate ffmpeg install is needed; faster-whisper decodes browser audio itself.

## 2. Run

```bash
uvicorn main:app --port 8000
```

The first start downloads the Whisper model (about 470 MB for `small`). Then open
**http://localhost:8000**. The badge in the top right turns green when the model is ready.

Use `localhost` (or HTTPS). Browsers block microphone access on plain `http://` addresses
that aren't localhost.

## 3. Use

1. Choose **Doctor** or **Patient**.
2. Press **Record**, speak in Urdu, press **Stop**.
3. The English text appears in that person's box after a few seconds.
4. With "Switch speaker after each turn" on, the selector flips automatically, so a
   conversation is just: Record, Stop, Record, Stop...

Both boxes are editable, so the doctor can correct anything before downloading.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`. Larger is more accurate and slower. |
| `WHISPER_DEVICE` | `auto` | `cpu` or `cuda` |
| `WHISPER_COMPUTE` | `auto` | e.g. `int8` (CPU), `float16` (GPU) |
| `OLLAMA_MODEL` | *(off)* | Turns on LLM cleanup of medical wording, e.g. `qwen2.5:7b` |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs |
| `PROMPT_TERM_COUNT` | `60` | How many terms from `terms.txt` are given to Whisper as a hint |

Example (macOS/Linux):

```bash
WHISPER_MODEL=large-v3 WHISPER_DEVICE=cuda uvicorn main:app --port 8000
```

Example (Windows PowerShell):

```powershell
$env:WHISPER_MODEL="medium"; uvicorn main:app --port 8000
```

### Recommended models

| Hardware | Model |
|---|---|
| Laptop, CPU only | `small` (fast) or `medium` (better, slower) |
| NVIDIA GPU, 6 GB+ | `large-v3` with `WHISPER_DEVICE=cuda` |

### Optional: LLM cleanup for medical terms

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:7b
OLLAMA_MODEL=qwen2.5:7b uvicorn main:app --port 8000
```

The LLM only corrects terminology and obvious translation errors. If Ollama isn't
running, the app silently falls back to Whisper's output.

## Improving medical accuracy

- Edit `terms.txt`. Put the terms you hear most at the top; those are used as a hint
  to Whisper. Every term in the file is used for the "detected" list.
- Use a bigger model. Urdu quality improves a lot from `small` to `large-v3`.
- Record one short turn at a time (a sentence or a few). Very long clips are less accurate.
- Use a decent microphone and a quiet room.

## Why you pick the speaker instead of auto-detection

Automatic speaker separation (diarization) is unreliable on a single microphone with short
turns and noise, and it can silently put a patient's words in the doctor's box. Choosing
the speaker per recording is far more dependable.

## Important

Machine translation can get medicines, doses and durations wrong. Treat the output as a
draft that a clinician reviews, not as a medical record. Don't upload real patient data to
any server you don't control; here everything runs on your own machine.
