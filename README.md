# EpsteinGPT

EpsteinGPT is a single-file, **epistemically-aware, multimodal LLM system** that can:

- Continuously ingest public documents (PDF, HTML, text) from government and archive sites
- Deduplicate content semantically using FAISS and Sentence Transformers
- Train a GPT-Neo–based language model with a custom epistemic loss (authority + provenance)
- Use **real multimodal encoders** (images, audio, video) for conditioning
- Expose an **OpenAI-style chat API** via FastAPI
- Provide a local REPL for interactive use
- Log all runs, steps, docs, and events in SQLite for observability

This repo is centered around a single Python script: `EpsteinGPT.py`.

---

## Features

- **Model & Training**
  - Backbone: `EleutherAI/gpt-neo-2.7B` (configurable)
  - Mixed-precision (AMP) training on GPU with automatic GradScaler
  - Custom **epistemic loss** combining:
    - Cross-entropy
    - Authority/provenance regularization via DAG-style information measures
  - Checkpointing to `./checkpoints` with disk-usage–aware pruning

- **Data Ingestion**
  - Periodic crawl of configured public archives (e.g., JFK files, OIG reports, GovernmentAttic)
  - Support for `.pdf`, `.txt`, `.html` (extensible)
  - PDF text extraction with PyPDF2, OCR fallback with pdf2image + Tesseract
  - Semantic deduplication using Sentence Transformers + FAISS (cosine similarity on normalized embeddings)
  - Per-document metadata store (JSON + JSONL log) with simple authority heuristics

- **Multimodal Encoders**
  - **Images:** CLIP (`openai/clip-vit-base-patch32`) for image embeddings
  - **Audio:** Whisper (`openai/whisper-small`) for:
    - Audio embeddings (mean-pooled encoder state)
    - Optional transcript text
  - **Video:** Frame sampling (~1 frame/sec) + CLIP image encoder; averaged embeddings

- **Multimodal Fusion**
  - `MultimodalPrefix` module:
    - Projects image/audio/video embeddings into LM hidden space via linear layers
    - Adds a fused prefix vector to token embeddings before generation
  - Enables conditioning the LM on multimodal context in a lightweight, extensible way

- **Storage & Observability**
  - SQLite database (`./epsteingpt.db`) with tables for:
    - `runs`, `steps`, `docs`, `events`
  - Metadata store with:
    - Buffered in-memory dict
    - JSONL append log
    - Periodic JSON snapshots
  - Supervisor thread:
    - Logs latest loss
    - Monitors disk usage, prunes old checkpoints
    - Logs GPU memory usage and triggers CUDA cache cleanup

- **Interfaces**
  - **REPL:** terminal-based interactive shell with simple commands:
    - `/status`, `/runs`, `/pause`, `/resume`, `/help`
  - **HTTP API:** FastAPI app exposing:
    - `POST /chat/completions` — OpenAI-style chat endpoint
  - Single shared model instance for:
    - Training (AMP)
    - REPL generation
    - API chat completion

---

## Requirements

Python 3.10+ is recommended.

### Python packages

Install via pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install transformers sentence-transformers faiss-cpu
pip install fastapi uvicorn[standard]
pip install pillow pytesseract pdf2image PyPDF2 beautifulsoup4 requests
pip install opencv-python
```

You also need:

- **Tesseract OCR** installed and available on PATH for image/PDF OCR
- **poppler** for `pdf2image` (platform-specific)
- **ffmpeg** is recommended for robust video decoding (OpenCV uses system codecs)

> Adjust PyTorch and CUDA versions according to your system.

---

## Configuration

Core configuration lives in the `Config` class at the top of `EpsteinGPT.py`. Key fields:

- Model/device:
    - `base_model`: Hugging Face model name (default `EleutherAI/gpt-neo-2.7B`)
    - `device`: `"cuda"` or `"cpu"` (auto-selected)
- Data \& training:
    - `data_path`: folder containing ingested files (default `./epstein_files`)
    - `max_len`, `stride`: sequence length and training stride
    - `batch_size`, `epochs`, `lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`
- Epistemic loss:
    - `num_roots`, `lambda_reg`, `sigma_a`, `sigma_h`, `alpha_base`, `rho_base`, `total_steps_cap`
- Ingestion:
    - `ingestion_interval_sec`
    - `keywords`: filter terms for links
    - `public_archives`: base URLs to crawl
- Dedup:
    - `embedding_model_name`: SentenceTransformer model
    - `dedup_index_path`, `dedup_meta_path`, `dedup_threshold`
- Checkpoints \& logs:
    - `checkpoint_dir`, `checkpoint_every_steps`
    - `metadata_store_path`, `metadata_log_path`
    - `db_path`
- Supervisor:
    - `supervisor_interval_sec`
    - `max_disk_usage_ratio`
    - `max_checkpoints_to_keep`
- API:
    - `api_host`, `api_port`, `enable_api`

Edit this class to customize behavior.

---

## Running EpsteinGPT

Save the script as `EpsteinGPT.py`, then:

```bash
python EpsteinGPT.py
```

On startup, the system will:

1. Initialize SQLite and metadata store.
2. Start the ingestion thread to periodically fetch and index new documents.
3. Load the GPT-Neo model and tokenizer.
4. Initialize multimodal components:
    - `ImageEncoder`, `AudioEncoder`, `VideoEncoder`, `MultimodalPrefix`.
5. Load the latest successful checkpoint if available.
6. Start the AMP-enabled training loop in a background thread.
7. Start the supervisor thread.
8. Start the FastAPI server (if `enable_api=True`) on `api_host:api_port`.
9. Enter the interactive REPL in your terminal.

To stop, press `Ctrl+C` in the terminal; the script will flush metadata and close the FAISS deduper.

---

## Using the REPL

In the REPL:

- Type a natural-language prompt and press Enter to get a model response.
- Commands:
    - `/status` — latest run, step, loss, document count
    - `/runs` — recent runs
    - `/pause` — request training pause
    - `/resume` — resume training in a new thread
    - `/help` — show commands
    - `exit` or `quit` — terminate the program

Generation in the REPL uses the same model that training is updating, protected by a training lock and `torch.inference_mode()`.

---

## HTTP Chat API

The FastAPI app exposes:

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
        { "type": "text", "text": "You are EpsteinGPT, a helpful assistant." }
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

For each non-text part, the `ChatEngine`:

1. Downloads the referenced media to a temporary file.
2. Encodes it using `ImageEncoder`, `AudioEncoder`, or `VideoEncoder`.
3. Passes embeddings to `MultimodalPrefix`, which adjusts token embeddings.
4. Calls `model.generate` with the fused embeddings.

### Response shape

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
          { "type": "text", "text": "..." }
        ]
      },
      "finish_reason": "stop"
    }
  ]
}
```

This makes EpsteinGPT easy to integrate with any existing GUI or client that already talks to OpenAI-style chat endpoints.

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

In other words, it’s built to be skeptical and evidence-focused, not just fluent.

### How You Use It Day-to-Day

Once someone with technical skills sets it up, you can use EpsteinGPT in two main ways:

1. **Chat in a terminal window (REPL)**
    - You type a question, it types an answer.
    - You can also run simple commands to check its status (how much it has trained, how many documents it has ingested, etc.).
2. **Use it through a chat API**
    - Any app that can talk to an OpenAI-style chat API can talk to EpsteinGPT instead.
    - That means developers can plug it into web apps, dashboards, or custom investigation tools without changing much code.

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

## Extending EpsteinGPT

Some obvious next steps:

- **Richer multimodal training:** include image/audio/video embeddings directly in training objectives, not only inference-time conditioning.
- **Multimodal FAISS indices:** build separate indices for image/audio/video embeddings for cross-modal retrieval and deduplication.
- **Streaming responses:** adapt `/chat/completions` to support server-sent events (SSE) or chunked responses.
- **Security hardening:** sandbox ingestion, validate/limit media downloads, restrict remote URLs in production.
- **Deployment:** run under a process manager (e.g., systemd) or container orchestrator, use multiple Uvicorn workers, and separate training from serving if needed.

---

## License, Usage, and Disclaimer

This is a research / experimental system intended as a reference implementation of:

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

```
```

