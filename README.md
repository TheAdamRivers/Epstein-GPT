# EpsteinGPT

EpsteinGPT is a self-hosted, epistemic large language model stack built around an EleutherAI GPT‑Neo 2.7B base model with:

- Online ingestion of public documents (JFK archives, OIG reports, etc.)
- Semantic deduplication using SentenceTransformers + FAISS
- Epistemic regularization loss for training
- FastAPI HTTP API exposing an OpenAI‑style `/chat/completions` endpoint
- Optional multimodal conditioning (images, audio, video) via CLIP and Whisper
- A simple Streamlit chat UI that talks to the FastAPI backend

This README describes how to install, configure, and run the full stack locally, and how to interact with it from the browser or from code.

---

## What EpsteinGPT is (non‑technical overview)

For non‑technical users, EpsteinGPT is a **local AI assistant** that you run on your own machine. Instead of sending data to a third‑party service, the model, database, and all processing stay under your control.

EpsteinGPT continuously pulls in public documents (such as declassified archives and government reports), cleans and deduplicates them, and uses them to inform its responses. It is designed to be *epistemic*—focused on sources, uncertainty, and evidence rather than just generating fluent text.

You can:

- Ask questions from a web chat interface (Streamlit) in your browser.
- Use a programmatic API (FastAPI) from your own tools or scripts.
- Optionally enable image, audio, and video understanding for richer queries.

---

## Why this project matters

EpsteinGPT explores what it looks like to build a **vertically integrated, self‑hosted research assistant** instead of depending on opaque cloud models.

Key aspects of significance:

- **Data provenance and authority**: The system tracks sources, local citation counts, and an authority score per document, enabling reasoning about “where a claim came from” and how trustworthy it might be.
- **Continuous ingestion and learning**: Ingestion and training loops enable the model to adapt over time to new public records and leaks, instead of relying on static snapshots.
- **Epistemic loss**: A custom loss term encourages the model not just to predict text, but to balance consistency, disagreement, and authority among sources.
- **Multimodal architecture**: With CLIP, Whisper, and a multimodal prefix module, the design supports conditioning on images, audio, and video in the same latent space as text.
- **Privacy and autonomy**: Running locally (or on controlled infrastructure) gives organizations and individuals more control over sensitive workflows than typical SaaS AI tools.

---

## Features

- **Causal LM backbone**
  - EleutherAI `gpt-neo-2.7B` via Hugging Face Transformers.

- **Training & checkpoints**
  - Streaming and static dataset support (`EpsteinDataset`, `StreamingEpsteinDataset`).
  - Epistemic loss combining cross‑entropy with a regularization term.
  - Checkpointing every N steps with metadata logging to SQLite (`runs`, `steps` tables).

- **Ingestion pipeline**
  - Crawls configured public archives (`archives.gov`, `justice.gov`, `governmentattic.org`, etc.).
  - Extracts text from PDFs, images (OCR), JSON, and common formats.
  - Semantic deduplication with SentenceTransformers + FAISS index.
  - Document registry in SQLite (`docs` table with path, hash, timestamps).

- **Epistemic metadata and authority scoring**
  - Heuristics to estimate citation density (patterns like “v.”, “U.S.C.”, “case no.”).
  - Authority scoring by source type (gov, court, media, unknown) and citation statistics.

- **Multimodal encoders (optional)**
  - Image encoder: CLIP (`openai/clip-vit-base-patch32`).
  - Audio encoder: Whisper (`openai/whisper-small`).
  - Video encoder: CLIP on sampled frames.
  - `MultimodalPrefix` projecting embeddings into LM hidden space and fusing them as a conditioning prefix.

- **FastAPI backend**
  - `POST /chat/completions?session_id=default` with OpenAI‑style request/response models.
  - Session‑aware `ChatEngine` storing conversation history per session ID.
  - Optional debug metadata in responses (e.g., which modalities were used).

- **Streamlit frontend**
  - Chat UI using `st.chat_message` and `st.chat_input`.
  - Sends OpenAI‑style JSON payloads and displays multi‑part text responses.

---

## Requirements

- Python 3.10+ (recommended).
- CPU with sufficient RAM for a 2.7B parameter model; GPU strongly recommended for interactive latency.
- System dependencies:
  - Tesseract OCR (for `pytesseract`).
  - Ghostscript / poppler or equivalent for `pdf2image` (platform‑specific).

Python packages (install via `pip`):

```bash
pip install \
  torch transformers sentence-transformers faiss-cpu \
  fastapi uvicorn pydantic \
  streamlit requests \
  beautifulsoup4 PyPDF2 pdf2image pytesseract Pillow \
  torchaudio opencv-python
```

Adjust `faiss-cpu` to a GPU‑enabled FAISS build if desired.

---

## Project structure

Core files and directories:

- `EpsteinGPT.py` – initial single‑file implementation with API and training.
- `EpsteinGPTv2.py` – updated multimodal‑aware version with enhanced chat response structure and post‑processing.
- `app.py` – Streamlit chat frontend (see below).
- `.epsteinfiles/` – ingestion corpus directory (auto‑created).
- `.checkpoints/` – checkpoint files (`.pt`) written by the training manager.
- `.mediacache/` – cached embeddings for media to avoid recomputation.
- `.epsteingpt.db` – SQLite database for runs, steps, docs, and events.

Configuration is embedded in the `Config` class inside `EpsteinGPT.py` / `EpsteinGPTv2.py`.

---

## Configuration

Key `Config` fields (non‑exhaustive):

- **Model & training**
  - `basemodel = "EleutherAI/gpt-neo-2.7B"`
  - `device = "cuda" if torch.cuda.is_available() else "cpu"`
  - `maxlen`, `stride`, `batchsize`, `epochs`, `lr`, `warmupratio`, `weightdecay`, `maxgradnorm`

- **Epistemic loss**
  - `numroots`, `lambdareg`, `sigmaa`, `sigmah`, `alphabase`, `rhobase`, `totalstepscap`

- **Ingestion**
  - `ingestionintervalsec`
  - `keywords`
  - `publicarchives`
  - `maxnewdocspercycle`

- **Deduplication**
  - `embeddingmodelname`
  - `dedupindexpath`
  - `dedupmetapath`
  - `dedupthreshold`

- **Checkpoints & DB**
  - `checkpointdir`, `checkpointeverysteps`
  - `metadatastorepath`, `metadatalogpath`
  - `dbpath`

- **Supervisor**
  - `supervisorintervalsec`, `maxdiskusageratio`, `maxcheckpointstokeep`

- **API & UI behavior**
  - `apihost = "0.0.0.0"`
  - `apiport = 8000`
  - `enableapi = True`
  - `enablestreaming` (reserved)
  - `maxuploadmb`
  - `safemodedefault`

- **Media cache & answers**
  - `mediacachedir`, `reusecachedembeddings`
  - `maxsummarychars`, `maxsummarybullets`

Modify `Config` to fit your environment before running.

---

## Running the backend

Launch the full stack by running `EpsteinGPTv2.py`:

```bash
python EpsteinGPTv2.py
```

On startup, it will:

- Initialize the SQLite database and metadata store.
- Load the base model and tokenizer.
- Optionally restore from the latest good checkpoint.
- Start ingestion, training, supervisor, and API threads (if enabled in `Config`).

Console output example:

```text
EpsteinGPT ready. Type 'help' for commands, 'exit' to quit.
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### REPL commands

In the same terminal, you can interact with the local REPL:

- `status` – show last run ID, last step, latest loss, and doc count.
- `runs` – list recent runs with timestamps and status.
- `pause` – request training pause.
- `resume` – resume training in a new thread.
- `help` – list commands.
- Any other input – sent as a prompt directly to the model for an immediate response.

Note: first responses after startup may be slow while the model and supporting components warm up. Subsequent responses are typically faster.

---

## API: `/chat/completions`

### Request format

Endpoint:

```text
POST /chat/completions?session_id={session_id}
```

Body (JSON) conforms to `ChatRequest`:

```json
{
  "model": "epsteingpt",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are EpsteinGPT, an epistemic AI assistant."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hello"
        }
      ]
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": false
}
```

Rules:

- `messages` must be non‑empty.
- Each message:
  - `role` in `"user"`, `"assistant"`, `"system"`.
  - `content` is a list of parts.
- Text parts: `{ "type": "text", "text": "..." }`
- The last message’s `role` **must** be `"user"` or `"system"`; otherwise the server returns `400 Bad Request`.
- `session_id` controls which server‑side history buffer is used.

Example curl:

```bash
curl -X POST "http://localhost:8000/chat/completions?session_id=default" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"epsteingpt\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}],\"temperature\":0.7,\"top_p\":0.9,\"max_tokens\":64,\"stream\":false}"
```

### Response format

`ChatResponse` example:

```json
{
  "id": "chatcmpl-1739976000000",
  "object": "chat.completion",
  "created": 1739976000,
  "model": "epsteingpt",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "Hello! How can I help you today?"
          }
        ],
        "debug": {
          "image_encoded": false,
          "audio_encoded": false,
          "video_encoded": false
        }
      },
      "finish_reason": "stop"
    }
  ]
}
```

Typical client logic:

- Take `choices[0].message`.
- Iterate `message["content"]`.
- Concatenate `part["text"]` for all parts where `part["type"] == "text"`.

---

## Streamlit chat app (for non‑technical users)

For non‑technical users, the easiest way to use EpsteinGPT is via a simple chat interface in your browser.

### Step 1: Create `app.py`

Place this file in the same directory as `EpsteinGPTv2.py`:

```python
import json
import time
from typing import List, Dict, Any

import requests
import streamlit as st


API_URL = "http://localhost:8000/chat/completions"
SESSION_ID = "default"
REQUEST_TIMEOUT = 300  # seconds


def build_backend_messages(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    backend_messages: List[Dict[str, Any]] = []
    for msg in history:
        role = msg["role"]
        if role not in ("user", "assistant", "system"):
            role = "system"

        backend_messages.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"],
                    }
                ],
            }
        )
    return backend_messages


def call_backend(history: List[Dict[str, str]]) -> str:
    messages = build_backend_messages(history)

    if not messages or messages[-1]["role"] not in ("user", "system"):
        raise ValueError("Last message must be user or system for backend request.")

    payload: Dict[str, Any] = {
        "model": "epsteingpt",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": False,
    }

    response = requests.post(
        f"{API_URL}?session_id={SESSION_ID}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in backend response: {json.dumps(data)[:400]}")

    message = choices.get("message", {})
    parts = message.get("content", [])

    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            t = part.get("text", "")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())

    if not texts:
        return json.dumps(message, ensure_ascii=False)

    return "\n\n".join(texts)


st.set_page_config(page_title="EpsteinGPT", page_icon="🤖")
st.title("EpsteinGPT")

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"] if msg["role"] in ("user", "assistant") else "assistant"):
        st.markdown(msg["content"])

user_text = st.chat_input("Type a message and press Enter")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("EpsteinGPT is thinking..."):
            start = time.time()
            try:
                assistant_text = call_backend(st.session_state.messages)
                elapsed = time.time() - start

                st.markdown(assistant_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                st.caption(f"Response time: {elapsed:.2f} s")

            except requests.HTTPError as http_err:
                status = http_err.response.status_code
                try:
                    err_body = http_err.response.json()
                    err_text = json.dumps(err_body, ensure_ascii=False)
                except Exception:
                    err_text = http_err.response.text[:400]
                st.error(f"Backend HTTP {status}: {err_text}")

            except Exception as e:
                st.error(f"Client error: {e}")
```

### Step 2: Start the backend

In one terminal:

```bash
python EpsteinGPTv2.py
```

Wait until you see “Application startup complete” and the Uvicorn line with port 8000.

### Step 3: Start the chat app

In another terminal:

```bash
streamlit run app.py
```

Your browser will open (or you can navigate to `http://localhost:8501`). You should see a chat window titled “EpsteinGPT”. Type “hello” and press Enter. The first answer may take some time; later exchanges are usually faster.

---

## Multimodal usage (advanced)

The Chat API supports image, audio, and video parts using URL references.

- Image:

  ```json
  {
    "type": "image_url",
    "image_url": {
      "url": "https://example.com/image.png",
      "detail": "high"
    }
  }
  ```

- Audio:

  ```json
  {
    "type": "audio_url",
    "audio_url": {
      "url": "https://example.com/audio.wav"
    }
  }
  ```

- Video:

  ```json
  {
    "type": "video_url",
    "video_url": {
      "url": "https://example.com/video.mp4"
    }
  }
  ```

When present in the last user message, the backend will:

- Download the referenced media (up to `maxuploadmb` megabytes).
- Compute modality‑specific embeddings via `ImageEncoder`, `AudioEncoder`, or `VideoEncoder`.
- Fuse them with text via `MultimodalPrefix` and call `model.generate` with `inputs_embeds`.

This can increase compute and latency, so start with text‑only usage if you are on a limited machine.

---

## Ingestion and training internals (technical users)

- **Ingestion loop** (`continuous_ingestion_loop`):
  - Periodically fetches new URLs from configured archives.
  - Filters by extension and keywords.
  - Downloads, extracts text, deduplicates, and stores metadata plus content hash.

- **Semantic dedup** (`SemanticDeduper`):
  - Uses SentenceTransformers to embed text.
  - FAISS index for similarity.
  - Threshold‑based duplicate detection and incremental index writes.

- **Training managers** (`TrainingManager`, `TrainingManagerAMP`):
  - Use either standard or AMP training loops.
  - Sample source metadata to feed into the epistemic loss.
  - Write steps and checkpoints and update the `runs` table.

- **Supervisor** (`Supervisor`):
  - Monitors latest loss, disk usage, and GPU memory.
  - Enforces maximum checkpoint count and logs health events.

These components are enabled by default but can be selectively disabled by commenting out their thread starts at the bottom of `EpsteinGPTv2.py` for a lean inference‑only setup.

---

## Troubleshooting

- **Streamlit never returns (“EpsteinGPT is thinking...”)**
  - Check the backend logs for `POST /chat/completions`. If no request shows up, verify `API_URL` in `app.py`.
  - If the POST appears but there is no response, the model may still be warming up or `generate` is slow for your hardware; try reducing `max_tokens` in the client payload.

- **HTTP 400 from `/chat/completions`**
  - Ensure `messages` is non‑empty and that the last `role` is `"user"` or `"system"`.
  - Confirm each message has `content` as a list with at least one `"type": "text"` part containing non‑empty text.

- **High memory / disk usage**
  - Decrease batch size, `maxlen`, and `epochs` in `Config`.
  - Consider disabling training and/or ingestion threads if you only need inference.

- **OCR or PDF issues**
  - Verify system installs for `tesseract` and PDF rendering.
  - Check that `pdf2image` and `pytesseract` run successfully on sample files.

---

## Author

- **Author**: Adam Rivers  
- **Role**: CEO, SynthicSoft Labs  
- **Website**: https://synthicsoftlabs.com  

---

## Disclaimer

EpsteinGPT is an experimental research system. It is **not** a product, and it comes with **no warranty or guarantee of accuracy**.

- The model can hallucinate, misinterpret documents, and produce incorrect or misleading answers.
- Ingestion sources include public archives and reports; their presence in the corpus does not imply endorsement or verification.
- This stack is **not** intended for medical, legal, financial, or safety‑critical decision making.
- You are responsible for complying with all relevant laws, regulations, and third‑party model/data licenses when running or modifying this code.

Use EpsteinGPT at your own risk and always independently verify important information against primary sources.
