# EpsteinGPT

EpsteinGPT is a **research-focused, epistemically-aware, multimodal LLM system** for deep investigative work over large, messy corpora (declassified files, court records, government archives, etc.).

This repository currently contains **two engine variants**:

- `EpsteinGPT.py` — original single-file implementation (baseline engine)
- `EpsteinGPTv2.py` — upgraded engine with multimodal caching, improved API responses, and UX-oriented refinements
- `LICENSE` — project license

Most new users should start with **EpsteinGPTv2** unless you specifically want the original reference version.

---

## Which Engine Should I Use?

| File              | Description                                                                                   | Recommended for                                           |
|-------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `EpsteinGPT.py`   | Original single-file engine with ingestion, training, basic multimodal support, and API.     | Reading the original design, baselines, audits.          |
| `EpsteinGPTv2.py` | Enhanced engine with media caching, safer multimodal handling, answer post-processing, etc.  | Actual use, new deployments, UI integration.             |

If you’re unsure, use **`EpsteinGPTv2.py`**.

---

## EpsteinGPTv2 Overview

`EpsteinGPTv2.py` is a drop-in evolution of the original script that:

- Keeps the **epistemic training** and ingestion pipeline intact.
- Adds **robust multimodal handling** (images, audio, video) with:
  - Safer encoding paths (size checks, error handling).
  - On-disk **embedding cache** (`./media_cache`) to avoid recomputing for reused files.
- Enhances the **chat API**:
  - Adds summary and key points to responses.
  - Returns a debug block indicating which modalities were successfully encoded.
- Keeps the **local REPL** and **FastAPI** interface.
- Is designed to be wrapped by a simple web UI (e.g., Streamlit) for non-technical users.

The rest of this README describes **EpsteinGPTv2**. Where differences with the original exist, they’re called out explicitly.

---

## Feature Summary (v2)

- **Model & Training**
  - Backbone: `EleutherAI/gpt-neo-2.7B` (configurable).
  - Mixed-precision (AMP) training on GPU with automatic GradScaler.
  - Custom **epistemic loss** combining:
    - Cross-entropy.
    - Authority/provenance regularization via DAG-style information measures.
  - Checkpointing to `./checkpoints` with disk-usage–aware pruning.
  - Supervisor that monitors loss, disk usage, GPU memory, and prunes old checkpoints.

- **Data Ingestion**
  - Periodic crawl of configured public archives (e.g., JFK files, OIG reports, GovernmentAttic).
  - Support for `.pdf`, `.txt`, `.html` (extensible).
  - PDF text extraction with PyPDF2, OCR fallback with pdf2image + Tesseract.
  - Semantic deduplication using Sentence Transformers + FAISS.
  - Per-document metadata store (JSON + JSONL log) with simple authority heuristics.

- **Multimodal Encoders**
  - **Images:** CLIP (`openai/clip-vit-base-patch32`) for image embeddings.
  - **Audio:** Whisper (`openai/whisper-small`) for:
    - Audio embeddings (mean-pooled encoder state).
    - Optional transcript text.
  - **Video:** Frame sampling (~1 frame / 3s) + CLIP image encoder; averaged embeddings.

- **Multimodal Fusion**
  - `MultimodalPrefix` module:
    - Projects image/audio/video embeddings into LM hidden space via linear layers.
    - Adds a fused prefix vector to token embeddings before generation.
  - No special tokens or prompt markers required; fusion happens in embedding space.

- **Media Embedding Cache (v2 only)**
  - On-disk cache (`./media_cache`) keyed by SHA256 of file bytes.
  - Safe wrappers:
    - `safe_encode_image_bytes`.
    - `safe_encode_audio_bytes`.
    - `safe_encode_video_bytes`.
  - Size limits via `max_upload_mb`.
  - Graceful degradation when media can’t be processed (text-only reply + note).

- **Storage & Observability**
  - SQLite database (`./epsteingpt.db`) with:
    - `runs`, `steps`, `docs`, `events`.
  - Metadata store with:
    - Buffered in-memory dict.
    - JSONL append log.
    - Periodic JSON snapshots.
  - Supervisor that prints latest loss and performs housekeeping.

- **Interfaces**
  - **REPL:** terminal-based interactive shell with commands:
    - `/status`, `/runs`, `/pause`, `/resume`, `/help`.
  - **HTTP API:** FastAPI app exposing:
    - `POST /chat/completions` — OpenAI-style chat endpoint.
  - **Optional Web UI:** Streamlit front-end (`app.py`, not required) for drag-and-drop usage.
  - Single shared model instance for:
    - Training (AMP).
    - REPL generation.
    - API chat completion.

- **Answer Post-processing (v2 only)**
  - Takes raw model output and:
    - Preserves full text.
    - Produces a short summary.
    - Extracts a few bullet-style key points.
  - API responses include full answer, summary, and key points in separate content blocks.

---

## File Layout

In this repo you’ll typically see:

- `EpsteinGPT.py`  
  Original engine. Older, simpler pipeline. Kept for reference and backwards compatibility.

- `EpsteinGPTv2.py`  
  Upgraded engine. If you’re starting fresh, **use this**.

- `LICENSE`  
  License for this project.

- (Optional) `app.py`  
  A small Streamlit front-end that talks to the v2 HTTP API.

---

## Requirements (for EpsteinGPTv2)

Python 3.10+ is recommended.

### Python packages

Core engine:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install transformers sentence-transformers faiss-cpu
pip install fastapi uvicorn[standard]
pip install pillow pytesseract pdf2image PyPDF2 beautifulsoup4 requests
pip install opencv-python
pip install python-multipart
```

Optional web UI:

```bash
pip install streamlit
```

You also need:

- **Tesseract OCR** installed and available on PATH for image/PDF OCR.  
- **poppler** for `pdf2image` (platform-specific).  
- **ffmpeg** for robust video decoding (OpenCV uses system codecs).

---

## Configuration (v2)

`EpsteinGPTv2.py` defines a `Config` class at the top. Key fields:

- **Model / device**
  - `base_model`: Hugging Face model name (default `EleutherAI/gpt-neo-2.7B`).
  - `device`: `"cuda"` or `"cpu"`.

- **Data & training**
  - `data_path`: folder containing ingested files (default `./epstein_files`).
  - `max_len`, `stride`: sequence length and training stride.
  - `batch_size`, `epochs`, `lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`.

- **Epistemic loss**
  - `num_roots`, `lambda_reg`, `sigma_a`, `sigma_h`, `alpha_base`, `rho_base`, `total_steps_cap`.

- **Ingestion**
  - `ingestion_interval_sec`.
  - `keywords`: filter terms for links.
  - `public_archives`: base URLs to crawl.
  - `max_new_docs_per_cycle`.

- **Dedup**
  - `embedding_model_name`.
  - `dedup_index_path`, `dedup_meta_path`, `dedup_threshold`.

- **Checkpoints & logs**
  - `checkpoint_dir`, `checkpoint_every_steps`.
  - `metadata_store_path`, `metadata_log_path`.
  - `db_path`.

- **Supervisor**
  - `supervisor_interval_sec`.
  - `max_disk_usage_ratio`.
  - `max_checkpoints_to_keep`.

- **API**
  - `api_host`, `api_port`, `enable_api`.

- **UI / behaviour**
  - `enable_streaming` (reserved for future WebSocket streaming).
  - `max_upload_mb`: max media size for backend processing.
  - `safe_mode_default`: default stance for “cautious” answers.

- **Media cache**
  - `media_cache_dir`.
  - `reuse_cached_embeddings`.

- **Answer post-processing**
  - `max_summary_chars`.
  - `max_summary_bullets`.

`EpsteinGPT.py` uses a similar but slightly smaller config surface (no media cache and post-processing).

---

## Running the Engines

### Original Engine (`EpsteinGPT.py`)

If you want to run the original version for reference:

```bash
python EpsteinGPT.py
```

You’ll get:

- Ingestion pipeline.
- Training loop.
- Basic multimodal support.
- REPL.
- FastAPI `/chat/completions` (simpler response shape, no summary/bullets/debug).

### Upgraded Engine (`EpsteinGPTv2.py`, recommended)

For the enhanced pipeline:

```bash
python EpsteinGPTv2.py
```

On startup (v2), the system:

1. Initializes SQLite and metadata store.
2. Starts the ingestion thread.
3. Loads the GPT-Neo model and tokenizer.
4. Initializes multimodal encoders and `MultimodalPrefix`.
5. Loads the latest successful checkpoint if available.
6. Starts the AMP-enabled training loop.
7. Starts the supervisor thread.
8. Starts the FastAPI server (if `enable_api=True`) on `api_host:api_port`.
9. Enters the interactive REPL.

Stopping: `Ctrl+C` in the terminal.

---

## Using the Local REPL (v2)

In the REPL:

- Type a natural-language prompt and press Enter to get a model response.
- Commands:
  - `/status` — latest run, step, loss, document count.
  - `/runs` — recent runs.
  - `/pause` — request training pause.
  - `/resume` — resume training in a new thread.
  - `/help` — show commands.
  - `exit` or `quit` — terminate the program.

Generation in the REPL uses the same model that training is updating, protected by a training lock and `torch.inference_mode()`.

---

## HTTP Chat API (v2)

### Endpoint

`POST /chat/completions`

### Request body (OpenAI-style)

```json
{
  "model": "epsteingpt",
  "messages": [
    {
      "role": "system",
      "content": [
        { "type": "text", "text": "You are EpsteinGPT, a factual, source-aware assistant." }
      ]
    },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Summarize what you know about declassified files." },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.png",
            "detail": "high"
          }
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

Supported content parts:

- `{"type": "text", "text": "..."}`  
- `{"type": "image_url", "image_url": {"url": "https://...", "detail": "high"}}`  
- `{"type": "audio_url", "audio_url": {"url": "https://..."}}`  
- `{"type": "video_url", "video_url": {"url": "https://..."}}`

For each non-text part, the v2 `ChatEngine`:

1. Downloads the referenced media bytes.
2. Validates size against `max_upload_mb`.
3. Uses the safe encoder wrappers:
   - `safe_encode_image_bytes`.
   - `safe_encode_audio_bytes`.
   - `safe_encode_video_bytes`.
4. Passes embeddings to `MultimodalPrefix`, which adjusts token embeddings.
5. Calls `model.generate` with the fused embeddings.

### Response shape (v2)

```json
{
  "id": "chatcmpl-1739999999999",
  "object": "chat.completion",
  "created": 1739999999,
  "model": "EleutherAI/gpt-neo-2.7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          { "type": "text", "text": "<full answer>" },
          { "type": "text", "text": "Summary:\n<short summary>" },
          { "type": "text", "text": "Key points:\n- bullet 1\n- bullet 2" }
        ],
        "debug": {
          "image_encoded": true,
          "audio_encoded": false,
          "video_encoded": false
        }
      },
      "finish_reason": "stop"
    }
  ]
}
```

If some attached media cannot be processed, an extra text part is appended:

```text
Note: Some attached files could not be processed. Try a different format or a shorter clip.
```

The original engine (`EpsteinGPT.py`) uses the same endpoint but returns a simpler text-only content block.

---

## Optional Web UI (v2)

If you add `app.py` as:

```python
import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat/completions"

st.set_page_config(page_title="EpsteinGPT", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("EpsteinGPT (v2)")

with st.sidebar:
    st.header("Session")
    safe_mode = st.checkbox("Safe mode (tone down sensitive content)", value=False)
    if st.button("Clear session"):
        st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        for media in m.get("media", []):
            if media["type"] == "image":
                st.image(media["data"])
            elif media["type"] == "audio":
                st.audio(media["data"])
            elif media["type"] == "video":
                st.video(media["data"])

st.markdown("### Attach media (optional)")
up_images = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
up_audio  = st.file_uploader("Audio", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
up_video  = st.file_uploader("Video", type=["mp4", "mov", "mkv"], accept_multiple_files=True)

user_text = st.chat_input("Ask EpsteinGPT anything about your data...")

if user_text:
    media_preview = []
    with st.chat_message("user"):
        st.markdown(user_text)
        for f in up_images or []:
            st.image(f)
            media_preview.append({"type": "image", "data": f})
        for f in up_audio or []:
            st.audio(f)
            media_preview.append({"type": "audio", "data": f})
        for f in up_video or []:
            st.video(f)
            media_preview.append({"type": "video", "data": f})

    st.session_state.messages.append({"role": "user", "text": user_text, "media": media_preview})

    content = [{"type": "text", "text": user_text}]
    if safe_mode:
        content.insert(0, {
            "type": "text",
            "text": "Answer cautiously and avoid sensational language."
        })

    payload = {
        "model": "epsteingpt",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are EpsteinGPT, a factual, source-aware assistant."}
                ],
            },
            {
                "role": "user",
                "content": content
            },
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False,
    }

    with st.spinner("Thinking..."):
        r = requests.post(API_URL, json=payload, timeout=300)
        r.raise_for_status()
        resp = r.json()
        parts = resp["choices"]["message"]["content"]
        raw = parts["text"]
        summary = ""
        bullets = ""
        if len(parts) > 1:
            summary = parts["text"] [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/150438246/4e7c4877-fb74-4abb-ae1e-5578907ba14b/paste.txt)
        if len(parts) > 2:
            bullets = parts["text"] [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/150438246/bea6e700-7764-4ccf-a844-600d5ad2c9b6/paste.txt)

    with st.chat_message("assistant"):
        st.markdown(raw)
        if summary:
            with st.expander("Summary"):
                st.markdown(summary)
        if bullets:
            with st.expander("Key points"):
                st.markdown(bullets)

    st.session_state.messages.append({"role": "assistant", "text": raw, "media": []})
```

Run:

```bash
streamlit run app.py
```

This gives you:

- Chat window with history.
- Drag-and-drop media inputs.
- Summary and key-point expanders.
- Safe-mode toggle for softer answers.

---

## Non-Technical Overview

If you’re not a programmer, here’s what EpsteinGPT actually does for you in plain language.

### What EpsteinGPT Is

EpsteinGPT is like a **research assistant that never gets tired**.  
You give it a topic or a question, and it:

- Hunts through large collections of documents you’ve fed it (like PDFs, web archives, or text files).
- Reads and remembers important details.
- Uses that knowledge to answer your questions as a conversational assistant.

You don’t need to understand how the AI is built to use it. Once it’s running, you can talk to it like you would to any other modern AI chatbot.

### What Makes It Different

Most chatbots just “talk”; EpsteinGPT is designed to **investigate**:

- It prefers **official sources** (like court records and government archives) over random blogs or social media.
- It keeps track of where its information comes from and how trustworthy each document seems.
- It can process **text, images, audio, and video**, not just plain text.
- It tries to present answers with a short summary and key points, so dense material is easier to digest.

In other words, it’s built to be skeptical and evidence-focused, not just fluent.

### How You Use It Day-to-Day

Once someone with technical skills sets it up, you can use EpsteinGPT in two main ways:

1. **Chat in a terminal window (REPL)**  
   - You type a question, it types an answer.  
   - You can also run simple commands to check its status (how much it has trained, how many documents it has ingested, etc.).

2. **Use it through a web UI or chat API**  
   - A simple web app can let you drag-and-drop images, audio, or video and ask questions about them.
   - Any app that can talk to an OpenAI-style chat API can talk to EpsteinGPT instead.

### What It Can Help You Do

For non-technical investigators, journalists, or researchers, EpsteinGPT can:

- Scan huge document dumps and help you **find connections** (names, places, dates, cases) that are easy to miss.
- Summarize long reports into something you can read in a few minutes.
- Cross-reference text with images, audio clips, or videos to build a fuller picture.
- Keep working in the background—downloading, cleaning, deduplicating, and learning from new material.

Think of it as a **persistent, detail-obsessed assistant** that keeps your research organized and helps you ask better questions.

### What You Still Need a Human For

EpsteinGPT doesn’t replace human judgment:

- It can surface patterns and evidence, but **you** decide what is meaningful or actionable.
- It can’t guarantee every source is honest or every document is complete.
- It’s a tool for **augmenting** your investigation, not a final authority.

If you can type a question into a search bar, you can use EpsteinGPT—  
the difference is that instead of just listing links, it works through your data with the patience and thoroughness of a full-time research team.

---

## What Is The Significance Of EpsteinGPT?

> The significance of this project lies in its ability to act as a tireless, objective digital detective that can process mountains of complex information faster and more reliably than a human team.  
>  
> Here is why this project is important and what makes it significant in non-technical terms:  
>  
> **1. It Finds the "Needle in the Haystack"**  
> The sheer volume of declassified documents, court filings, and audio recordings related to high-profile cases is overwhelming. This project is significant because it automatically hunts for this data across the web, reads through it all, and organizes it into a single, searchable brain. It ensures that no detail is missed simply because a human didn't have the time to read page 500 of a 1,000-page document.  
>  
> **2. It Prioritizes Facts Over Opinions**  
> One of the most important features is its "Truth Filter" (the Epistemic Loss function). In an era of misinformation, this system is programmed to:  
>  
> - Check the Source: It gives much more importance to an official court transcript or a government archive than to a news article or a blog post.  
> - Look for Evidence: It looks for legal citations (like specific law codes or case numbers) to decide how "authoritative" a document is.  
>  
> This means the AI is less likely to be "brainwashed" by rumors and more likely to stick to verifiable facts.  
>  
> **3. It "Sees" and "Hears" the Full Story**  
> Most AI systems only understand text. This project is significant because it has multi-sensory intelligence:  
>  
> - It can listen to audio recordings of depositions or hearings.  
> - It can analyze photos or video evidence.  
>  
> This allows the AI to connect a name mentioned in a court document to a person seen in a video or heard in an audio clip, creating a much more complete picture of an investigation.  
>  
> **4. It is Designed for Independence (Sovereignty)**  
> The code is built to run on your own hardware rather than relying on a big tech company's servers.  
>  
> - No Censorship: Because it is "sovereign," it can process sensitive topics without being shut down or filtered by external companies.  
> - Self-Managing: It even has its own "Janitor" (the Supervisor) that manages its own storage space, ensuring it doesn't crash your computer as it grows smarter.  
>  
> **Summary: Why it Matters**  
> The EpsteinGPT project is important because it provides a way to conduct deep, automated investigative research that is grounded in high-quality evidence. It levels the playing field, giving independent researchers a powerful tool that can process information with the speed of a machine but the skepticism of a veteran investigator.

---

## Author

Developed by **Adam Rivers** — CEO, **Synthicsoft Labs**.

---

## License, Usage, and Disclaimer

See `LICENSE` in this repository for licensing details.

EpsteinGPT is a research / experimental system intended as a reference implementation of:

- epistemic training concepts, and  
- multimodal conditioning on top of a legacy LM.

If you deploy it in production, you are responsible for:

- securing the ingestion pipeline,  
- complying with licenses and terms for all models and datasets,  
- ensuring resource usage (GPU/CPU/memory) fits your environment.

**Disclaimer**

- EpsteinGPT is provided **as-is**, with no warranty or guarantee of accuracy, reliability, or fitness for any particular purpose.  
- The system may surface incorrect, incomplete, or outdated information and should **not** be treated as a definitive source of truth.  
- It may process and reason over sensitive or controversial material. How you use its outputs—legally, ethically, and operationally—is entirely your responsibility.  
- The author and contributors are not liable for any decisions, actions, or consequences arising from the use of this software or any models, data, or services it relies on.
