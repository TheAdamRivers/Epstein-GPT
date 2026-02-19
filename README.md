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

***

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

***

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

***

## Configuration

Core configuration lives in the `Config` class at the top of `EpsteinGPT.py`. Key fields:

- Model/device:
  - `base_model`: Hugging Face model name (default `EleutherAI/gpt-neo-2.7B`)
  - `device`: `"cuda"` or `"cpu"` (auto-selected)
- Data & training:
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
- Checkpoints & logs:
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

***

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

***

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

***

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

***

## Multimodal Details

### Image

- Model: CLIP (`openai/clip-vit-base-patch32`)
- Pipeline:
  - Load image with PIL
  - Preprocess with `CLIPProcessor`
  - Get features with `CLIPModel.get_image_features`
  - Normalize to unit length

### Audio

- Model: Whisper (`openai/whisper-small`)
- Pipeline:
  - Load audio with `torchaudio.load`
  - Preprocess with `WhisperProcessor`
  - Use encoder output:
    - Mean-pool hidden states for embeddings
    - Optionally decode with `generate()` for transcript

### Video

- Pipeline:
  - Read video via OpenCV
  - Sample ~1 frame per second
  - Convert frames to RGB PIL
  - Encode each frame with CLIP image encoder
  - Average embeddings to form a single video vector

### Fusion

- `MultimodalPrefix` uses linear projections from each modality embedding space into the LM hidden size.
- A single fused vector is added as a bias to token embeddings for all positions.
- This is a light-weight, inference-time conditioning mechanism; you can later include it in training.

***

## Data & Training Flow

1. **Ingestion thread**
   - Crawls configured archives periodically.
   - Downloads candidate files, extracts text, computes hash and embeddings.
   - Uses FAISS to drop near-duplicates.
   - Writes metadata and records documents in SQLite.

2. **Training**
   - Streams text from `data_path` via iterable dataset.
   - For each batch:
     - Runs LM forward pass.
     - Samples synthetic epistemic variables (for now) alongside stored metadata priors.
     - Computes epistemic loss and backprop with AMP (if CUDA).
     - Periodically saves checkpoints and logs step metrics to SQLite.

3. **Supervision**
   - Monitors:
     - Latest loss
     - Disk usage, pruning checkpoints beyond configured count
     - GPU memory usage
   - Can be extended for alerts/telemetry.

***

## Extending EpsteinGPT

Some obvious next steps:

- **Richer multimodal training:** include image/audio/video embeddings directly in training objectives, not only inference-time conditioning.
- **Multimodal FAISS indices:** build separate indices for image/audio/video embeddings for cross-modal retrieval and deduplication.
- **Streaming responses:** adapt `/chat/completions` to support server-sent events (SSE) or chunked responses.
- **Security hardening:** sandbox ingestion, validate/limit media downloads, restrict remote URLs in production.
- **Deployment:** run under a process manager (e.g., systemd) or container orchestrator, use multiple Uvicorn workers, and separate training from serving if needed.

***

## License and Usage

This is a research / experimental system intended as a reference implementation of:

- epistemic training concepts, and
- multimodal conditioning on top of a legacy LM.

If you deploy it in production, you are responsible for:

- securing the ingestion pipeline,
- complying with licenses and terms for all models and datasets,
- ensuring resource usage (GPU/CPU/memory) fits your environment.
