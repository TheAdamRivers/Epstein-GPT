# EpsteinGPT

EpsteinGPT is a **research-focused, epistemically-aware, multimodal LLM system** for deep investigative work over large, messy corpora (declassified files, court records, government archives, etc.).

This repo currently contains **two engine variants**:

- `EpsteinGPT.py` — original single-file implementation (baseline engine)
- `EpsteinGPTv2.py` — upgraded engine with multimodal caching, improved API responses, and UX-oriented refinements
- `LICENSE` — license for this project

Most new users should start with **EpsteinGPTv2** unless you specifically want the original reference version.

---

## Which File Should I Use?

| File               | Description                                                                                  | Recommended for                                           |
|--------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `EpsteinGPT.py`    | Original single-file engine with ingestion, training, basic multimodal support, and API.    | Reading the original design, minimal baselines, audits.  |
| `EpsteinGPTv2.py`  | Enhanced engine with media caching, safer multimodal handling, answer post-processing, etc. | Actual use, new deployments, UI integration.             |

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

## Features (v2)

- **Model & Training**
  - Backbone: `EleutherAI/gpt-neo-2.7B` (configurable)
  - Mixed-precision (AMP) training on GPU with automatic GradScaler
  - Custom **epistemic loss** combining:
    - Cross-entropy
    - Authority/provenance regularization via DAG-style information measures
  - Checkpointing to `./checkpoints` with disk-usage–aware pruning
  - Supervisor that monitors loss, disk usage, GPU memory, and prunes old checkpoints

- **Data Ingestion**
  - Periodic crawl of configured public archives (e.g., JFK files, OIG reports, GovernmentAttic)
  - Support for `.pdf`, `.txt`, `.html` (extensible)
  - PDF text extraction with PyPDF2, OCR fallback with pdf2image + Tesseract
  - Semantic deduplication using Sentence Transformers + FAISS
  - Per-document metadata store (JSON + JSONL log) with simple authority heuristics

- **Multimodal Encoders**
  - **Images:** CLIP (`openai/clip-vit-base-patch32`) for image embeddings
  - **Audio:** Whisper (`openai/whisper-small`) for:
    - Audio embeddings (mean-pooled encoder state)
    - Optional transcript text
  - **Video:** Frame sampling (~1 frame / 3s) + CLIP image encoder; averaged embeddings

- **Multimodal Fusion**
  - `MultimodalPrefix` module:
    - Projects image/audio/video embeddings into LM hidden space via linear layers
    - Adds a fused prefix vector to token embeddings before generation
  - No special tokens or prompt markers required; fusion happens in embedding space.

- **Media Embedding Cache (v2 only)**
  - On-disk cache (`./media_cache`) keyed by SHA256 of file bytes
  - Safe wrappers:
    - `safe_encode_image_bytes`
    - `safe_encode_audio_bytes`
    - `safe_encode_video_bytes`
  - Size limits via `max_upload_mb`
  - Graceful degradation when media can’t be processed (text-only reply + note)

- **Storage & Observability**
  - SQLite database (`./epsteingpt.db`) with:
    - `runs`, `steps`, `docs`, `events`
  - Metadata store with:
    - Buffered in-memory dict
    - JSONL append log
    - Periodic JSON snapshots
  - Supervisor that prints latest loss and performs housekeeping

- **Interfaces**
  - **REPL:** terminal-based interactive shell with commands:
    - `/status`, `/runs`, `/pause`, `/resume`, `/help`
  - **HTTP API:** FastAPI app exposing:
    - `POST /chat/completions` — OpenAI-style chat endpoint
  - **Optional Web UI:** Streamlit front-end (`app.py`, not required) for drag-and-drop usage
  - Single shared model instance for:
    - Training (AMP)
    - REPL generation
    - API chat completion

- **Answer Post-processing (v2 only)**
  - Takes raw model output and:
    - Preserves full text
    - Produces a short summary
    - Extracts a few bullet-style key points
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

## Requirements (for v2)

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

- **Tesseract OCR** installed and available on PATH for image/PDF OCR  
- **poppler** for `pdf2image` (platform-specific)  
- **ffmpeg** for robust video decoding (OpenCV uses system codecs)

---

## Configuration (v2)

`EpsteinGPTv2.py` defines a `Config` class at the top. Key fields:

- Model/device:
  - `base_model`, `device`
- Data & training:
  - `data_path`, `max_len`, `stride`, `batch_size`, `epochs`, `lr`, `warmup_ratio`, `weight_decay`, `max_grad_norm`
- Epistemic loss:
  - `num_roots`, `lambda_reg`, `sigma_a`, `sigma_h`, `alpha_base`, `rho_base`, `total_steps_cap`
- Ingestion:
  - `ingestion_interval_sec`, `keywords`, `public_archives`, `max_new_docs_per_cycle`
- Dedup:
  - `embedding_model_name`, `dedup_index_path`, `dedup_meta_path`, `dedup_threshold`
- Checkpoints & logs:
  - `checkpoint_dir`, `checkpoint_every_steps`, `metadata_store_path`, `metadata_log_path`, `db_path`
- Supervisor:
  - `supervisor_interval_sec`, `max_disk_usage_ratio`, `max_checkpoints_to_keep`
- API:
  - `api_host`, `api_port`, `enable_api`
- UI / behaviour:
  - `enable_streaming` (reserved), `max_upload_mb`, `safe_mode_default`
- Media cache:
  - `media_cache_dir`, `reuse_cached_embeddings`
- Answer post-processing:
  - `max_summary_chars`, `max_summary_bullets`

`EpsteinGPT.py` uses a similar but slightly smaller config surface (no media cache and post-processing).

---

## Running the Original vs v2

### Original engine

If you want to run the original version for reference:

```bash
python EpsteinGPT.py
```

You’ll get:

- Ingestion pipeline
- Training loop
- Basic multimodal support
- REPL
- FastAPI `/chat/completions` (simpler response shape, no summary/bullets/debug)

### Upgraded engine (recommended)

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

## HTTP Chat API (v2)

Same endpoint for both engines, but v2 returns richer content.

### Endpoint

`POST /chat/completions`

### Request body (v2)

See the previous section for a full example; the structure is OpenAI-style with multimodal content parts:

- `text`
- `image_url`
- `audio_url`
- `video_url`

### Response (v2)

You receive:

- Full answer
- Summary
- Key points
- Debug info indicating which modalities were successfully encoded

The original engine (`EpsteinGPT.py`) returns a simpler message payload (single text content, no summary/bullets/debug).

---

## Optional Web UI (v2)

If you add `app.py` as described and run:

```bash
streamlit run app.py
```

it will talk to the **v2** API by default, but you can point it at the original engine’s port if you want.

---

## Non-Technical Overview

*(applies to both original and v2, but v2 is nicer to use)*

[Same non-technical overview text as before, describing EpsteinGPT as a tireless research assistant, why it matters, how it helps, and the limits of its judgment.]

---

## Author

Developed by **Adam Rivers** — CEO, **Synthicsoft Labs**.

---

## License, Usage, and Disclaimer

See `LICENSE` in this repository for licensing details.

EpsteinGPT is a research / experimental system. If you deploy it in production, you are responsible for security, compliance, and resource management. The software is provided **as-is**, without warranty.
