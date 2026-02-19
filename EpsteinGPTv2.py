import os
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Iterator, Optional, Any

import sqlite3
import shutil
import hashlib
import traceback

import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer

# New: API / multimodal imports
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from transformers import CLIPProcessor, CLIPModel, WhisperProcessor, WhisperModel
import torchaudio
import cv2
import torch.nn as nn

# =========================================================
# 0. Global Performance Tweaks
# =========================================================

torch.backends.cudnn.benchmark = torch.cuda.is_available()

# =========================================================
# 1. Configuration
# =========================================================


class Config:
    # Core model & training
    base_model = "EleutherAI/gpt-neo-2.7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "./epstein_files"

    max_len = 1024
    stride = 256
    batch_size = 2
    epochs = 3
    lr = 5e-5
    warmup_ratio = 0.03
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Epistemic loss
    num_roots = 7
    lambda_reg = 0.25
    sigma_a = 0.1
    sigma_h = 0.05
    alpha_base = 2.5
    rho_base = 2.0
    total_steps_cap = 1_000_000

    # Ingestion / crawl
    ingestion_interval_sec = 1800
    keywords = ["epstein", "declassified", "leak", "government", "files"]
    max_new_docs_per_cycle = 10
    public_archives = [
        "https://www.archives.gov/research/jfk",
        "https://www.justice.gov/oig/reports",
        "https://www.governmentattic.org",
    ]

    # Semantic dedup
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dedup_index_path = "./dedup_index.faiss"
    dedup_meta_path = "./dedup_meta.jsonl"
    dedup_threshold = 0.90

    # Checkpoints & metadata
    checkpoint_dir = "./checkpoints"
    checkpoint_every_steps = 500
    metadata_store_path = "./metadata_store.json"
    metadata_log_path = "./metadata_store.log.jsonl"
    db_path = "./epsteingpt.db"

    supervisor_interval_sec = 60
    max_disk_usage_ratio = 0.9
    max_checkpoints_to_keep = 10
    faiss_ivf_threshold = 1_000_000  # future use

    # API / chat
    api_host = "0.0.0.0"
    api_port = 8000
    enable_api = True

    # UI / API behaviour
    enable_streaming: bool = False          # reserved for WebSockets
    max_upload_mb: int = 64                 # backend media size limit
    safe_mode_default: bool = False

    # Media embedding cache
    media_cache_dir: str = "./media_cache"
    reuse_cached_embeddings: bool = True

    # Answer post-processing
    max_summary_chars: int = 400
    max_summary_bullets: int = 5


CFG = Config()
os.makedirs(CFG.data_path, exist_ok=True)
os.makedirs(CFG.checkpoint_dir, exist_ok=True)
os.makedirs(CFG.media_cache_dir, exist_ok=True)

# =========================================================
# 2. SQLite Persistence Layer
# =========================================================

_db_lock = threading.Lock()
_db_conn: Optional[sqlite3.Connection] = None


def get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(CFG.db_path, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
        with _db_conn:
            _db_conn.execute("PRAGMA journal_mode=WAL;")
            _db_conn.execute("PRAGMA synchronous=NORMAL;")
            _db_conn.execute("PRAGMA foreign_keys=ON;")
    return _db_conn


def init_db():
    conn = get_db()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                last_step INTEGER DEFAULT 0,
                last_ckpt_path TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                run_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                loss REAL,
                ce REAL,
                reg REAL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (run_id, step),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                path TEXT PRIMARY KEY,
                ingested_at REAL NOT NULL,
                hash TEXT,
                used_in_training INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                time REAL NOT NULL,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message_json TEXT NOT NULL
            );
            """
        )


def log_event(level: str, source: str, message: Dict):
    conn = get_db()
    payload = dict(message)
    if "traceback" not in payload:
        payload["traceback"] = traceback.format_exc()
    with _db_lock, conn:
        conn.execute(
            "INSERT INTO events(time, level, source, message_json) VALUES(?,?,?,?)",
            (time.time(), level, source, json.dumps(payload, ensure_ascii=False)),
        )


def log_run_start(config: Dict) -> int:
    conn = get_db()
    with _db_lock, conn:
        cur = conn.execute(
            "INSERT INTO runs(start_time, status, config_json) VALUES(?,?,?)",
            (time.time(), "running", json.dumps(config, ensure_ascii=False)),
        )
    return cur.lastrowid


def log_run_step(
    run_id: int,
    step: int,
    loss: float,
    ce: float,
    reg: float,
    ckpt_path: Optional[str],
):
    conn = get_db()
    with _db_lock, conn:
        conn.execute(
            "INSERT OR REPLACE INTO steps(run_id, step, loss, ce, reg, timestamp) "
            "VALUES(?,?,?,?,?,?)",
            (run_id, step, loss, ce, reg, time.time()),
        )
        conn.execute(
            "UPDATE runs SET last_step = ?, last_ckpt_path = ? WHERE id = ?",
            (step, ckpt_path, run_id),
        )


def log_run_end(run_id: int, status: str):
    conn = get_db()
    with _db_lock, conn:
        conn.execute(
            "UPDATE runs SET end_time = ?, status = ? WHERE id = ?",
            (time.time(), status, run_id),
        )


def get_last_good_checkpoint() -> Optional[Tuple[int, str]]:
    conn = get_db()
    with _db_lock, conn:
        row = conn.execute(
            "SELECT id, last_ckpt_path FROM runs "
            "WHERE status='success' AND last_ckpt_path IS NOT NULL "
            "ORDER BY end_time DESC LIMIT 1"
        ).fetchone()
    if row and row["last_ckpt_path"]:
        return row["id"], row["last_ckpt_path"]
    return None


def record_doc_ingested(path: Path, file_hash: str):
    conn = get_db()
    with _db_lock, conn:
        conn.execute(
            "INSERT OR REPLACE INTO docs(path, ingested_at, hash, used_in_training) "
            "VALUES(?,?,?,COALESCE((SELECT used_in_training FROM docs WHERE path=?),0))",
            (str(path.resolve()), time.time(), file_hash, str(path.resolve())),
        )

# =========================================================
# 3. Metadata Store (JSON + dict, batched + JSONL log)
# =========================================================

_metadata_lock = threading.Lock()
_metadata_store: Dict[str, Dict] = {}
_metadata_buffer: List[Tuple[str, Dict]] = []
_METADATA_BUFFER_FLUSH = 50
_METADATA_SNAPSHOT_INTERVAL = 1000


def load_metadata_store():
    global _metadata_store
    path = Path(CFG.metadata_store_path)
    if path.exists():
        try:
            _metadata_store = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _metadata_store = {}


def _flush_metadata_buffer():
    global _metadata_buffer
    if not _metadata_buffer:
        return
    with _metadata_lock:
        for key, meta in _metadata_buffer:
            _metadata_store[key] = meta
        with open(CFG.metadata_log_path, "a", encoding="utf-8") as f:
            for key, meta in _metadata_buffer:
                f.write(json.dumps({"key": key, "meta": meta}, ensure_ascii=False) + "\n")
        if len(_metadata_store) % _METADATA_SNAPSHOT_INTERVAL == 0:
            Path(CFG.metadata_store_path).write_text(
                json.dumps(_metadata_store, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _metadata_buffer = []


def infer_doc_type_from_path(path: Path) -> str:
    p = str(path).lower()
    if "gov" in p:
        return "gov"
    if "court" in p or "case" in p:
        return "court"
    if "news" in p or "media" in p:
        return "media"
    return "unknown"


def estimate_local_citation_count(text: str) -> int:
    patterns = ["v.", "vs.", "case no.", "u.s.c.", "§", "[", "]"]
    count = 0
    lower = text.lower()
    for pat in patterns:
        count += lower.count(pat.lower())
    return count


def update_metadata_for_file(path: Path, text: str, source_url: str | None = None):
    global _metadata_buffer
    key = str(path.resolve())
    doc_type = infer_doc_type_from_path(path)
    local_cites = estimate_local_citation_count(text)
    rank_score = 0.5
    meta = {
        "local_path": key,
        "source_url": source_url,
        "doc_type": doc_type,
        "local_citation_count": local_cites,
        "rank_score": rank_score,
        "timestamp": time.time(),
    }
    _metadata_buffer.append((key, meta))
    if len(_metadata_buffer) >= _METADATA_BUFFER_FLUSH:
        _flush_metadata_buffer()


def compute_authority(source_metadata: Dict) -> float:
    rank_score = float(source_metadata.get("rank_score", 0.0))
    local_cites = int(source_metadata.get("local_citation_count", 0))
    doc_type = source_metadata.get("doc_type", "unknown")
    type_boost = {
        "gov": 1.0,
        "court": 0.8,
        "media": 0.4,
        "unknown": 0.0,
    }.get(doc_type, 0.0)
    authority = np.tanh(np.log1p(local_cites) + rank_score + type_boost)
    return float(np.clip(authority, 0.0, 0.99))


def sample_source_metadatas(n: int) -> List[Dict]:
    with _metadata_lock:
        values = list(_metadata_store.values())
        if not values:
            return [
                {
                    "rank_score": 0.5 + float(np.random.rand()),
                    "local_citation_count": int(np.random.randint(0, 10)),
                    "doc_type": "unknown",
                }
                for _ in range(n)
            ]
        return [values[np.random.randint(0, len(values))] for _ in range(n)]

# =========================================================
# 4. Utility: Stable Text Hash + Media Cache Helpers
# =========================================================


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def media_cache_path(hash_hex: str, suffix: str) -> Path:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return Path(CFG.media_cache_dir) / f"{hash_hex}{safe_suffix}"


def load_cached_embedding(hash_hex: str) -> Optional[np.ndarray]:
    if not CFG.reuse_cached_embeddings:
        return None
    npy = media_cache_path(hash_hex, ".npy")
    if not npy.exists():
        return None
    try:
        return np.load(npy)
    except Exception:
        return None


def save_cached_embedding(hash_hex: str, emb: np.ndarray) -> None:
    try:
        npy = media_cache_path(hash_hex, ".npy")
        np.save(npy, emb.astype(np.float32))
    except Exception as e:
        log_event("WARN", "media_cache", {"msg": "save_failed", "hash": hash_hex, "error": str(e)})

# =========================================================
# 5. Multi-modal Extraction (core, text-centric)
# =========================================================


def extract_text_from_pdf(path: Path) -> str:
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception:
        text = ""
    if len(text.strip()) < 20:
        try:
            images = convert_from_path(str(path))
            for img in images:
                text += pytesseract.image_to_string(img)
        except Exception:
            pass
    return text


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            claims = data.get("claims", [])
            return "\n".join(c.get("text", "") for c in claims if c.get("text"))
        except Exception:
            return ""
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        try:
            return pytesseract.image_to_string(Image.open(path))
        except Exception:
            return ""
    return ""


def load_corpus(folder_path: str) -> List[str]:
    texts: List[str] = []
    for f in Path(folder_path).glob("**/*"):
        if f.is_file():
            t = extract_text(f)
            if t.strip():
                texts.append(t)
    return texts

# =========================================================
# 6. Semantic Dedup (Local, batched FAISS writes)
# =========================================================

class SemanticDeduper:
    def __init__(self, model_name: str, index_path: str, meta_path: str, threshold: float):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.meta_path = meta_path
        self.threshold = threshold
        self.lock = threading.Lock()
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.meta_f = open(self.meta_path, "a+", encoding="utf-8")
        self.meta_f.seek(0, os.SEEK_END)
        self._pending_adds = 0
        self._flush_every = 50

    def _embed(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        return emb.astype("float32")

    def is_duplicate(self, text: str) -> bool:
        emb = self._embed(text)
        with self.lock:
            if self.index.ntotal == 0:
                return False
            D, _ = self.index.search(emb.reshape(1, -1), k=1)
            sim = float(D[0][0])
            return sim >= self.threshold

    def _flush_index_if_needed(self, force: bool = False):
        with self.lock:
            if force or self._pending_adds >= self._flush_every:
                faiss.write_index(self.index, self.index_path)
                self._pending_adds = 0

    def add(self, text: str, meta: Dict):
        emb = self._embed(text)
        with self.lock:
            self.index.add(emb.reshape(1, -1))
            self.meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            self.meta_f.flush()
            self._pending_adds += 1
            self._flush_index_if_needed(force=False)

    def close(self):
        try:
            self._flush_index_if_needed(force=True)
            if hasattr(self, "meta_f") and not self.meta_f.closed:
                self.meta_f.close()
        except Exception:
            pass


DEDUP = SemanticDeduper(
    CFG.embedding_model_name,
    CFG.dedup_index_path,
    CFG.dedup_meta_path,
    CFG.dedup_threshold,
)

# =========================================================
# 7. Dataset & Streaming Dataset
# =========================================================


def chunk_ids(token_ids: List[int], max_len: int, stride: int) -> List[List[int]]:
    chunks: List[List[int]] = []
    i = 0
    while i < len(token_ids):
        window = token_ids[i : i + max_len]
        if len(window) < 8:
            break
        chunks.append(window)
        if i + max_len >= len(token_ids):
            break
        i += max_len - stride
    return chunks


class EpsteinDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int, stride: int):
        self.examples: List[List[int]] = []
        self.tokenizer = tokenizer
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            for chunk in chunk_ids(ids, max_len, stride):
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        enc = self.tokenizer(
            ids,
            truncation=True,
            max_length=CFG.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def build_dataloader(tokenizer):
    texts = load_corpus(CFG.data_path)
    dataset = EpsteinDataset(texts, tokenizer, CFG.max_len, CFG.stride)
    num_workers = 0 if CFG.device == "cpu" else 2
    pin_memory = CFG.device == "cuda"
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class StreamingEpsteinDataset(IterableDataset):
    def __init__(self, data_root: str, tokenizer, max_len: int, stride: int):
        super().__init__()
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

    def _iter_files(self) -> Iterator[Path]:
        for f in Path(self.data_root).glob("**/*"):
            if f.is_file():
                yield f

    def __iter__(self):
        try:
            for f in self._iter_files():
                text = extract_text(f)
                if not text or not text.strip():
                    continue
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                for chunk in chunk_ids(ids, self.max_len, self.stride):
                    enc = self.tokenizer(
                        chunk,
                        truncation=True,
                        max_length=self.max_len,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    attention_mask = enc["attention_mask"].squeeze(0)
                    yield input_ids, attention_mask
        except Exception as e:
            log_event("ERROR", "streaming_dataset", {"msg": "iter_error", "error": str(e)})


def build_streaming_dataloader(tokenizer) -> DataLoader:
    ds = StreamingEpsteinDataset(CFG.data_path, tokenizer, CFG.max_len, CFG.stride)
    num_workers = 0 if CFG.device == "cpu" else 2
    pin_memory = CFG.device == "cuda"
    persistent_workers = num_workers > 0
    return DataLoader(
        ds,
        batch_size=CFG.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

# =========================================================
# 8. Epistemic Functions
# =========================================================


def dag_information(root_probs, corr_matrix, modality_weights=None):
    eps = 1e-8
    p = root_probs.clamp(min=eps)
    if modality_weights is not None:
        p = p * modality_weights
    n = p.size(-1)
    simpson = (p ** 2).sum(dim=-1)
    mi_penalty = corr_matrix.sum(dim=(-2, -1)) / (n * n + eps)
    eff_support = 1.0 / (simpson + mi_penalty + eps)
    return (eff_support - 1.0) / (n - 1.0)


def authority_online(A_prior, model_agreement, sigma_a=0.1, beta=3.0):
    mu = torch.tanh(beta * (A_prior + model_agreement))
    dist = torch.distributions.Normal(mu, sigma_a)
    return dist.rsample()


def provenance_online(root_probs, model_agreement, corr_matrix, sigma_h=0.05):
    adjusted = root_probs + 0.5 * (model_agreement - 0.5)
    adjusted = torch.clamp(adjusted, 1e-8, 1.0)
    adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True)
    H_eff = dag_information(adjusted, corr_matrix)
    return H_eff, sigma_h


def imbalance(A, H, sigma_a, sigma_h):
    kappa = torch.exp(-(sigma_a + sigma_h))
    return kappa * (A - H)


def adaptive_reg(imb, step, total_steps=1_000_000, alpha_base=2.5, rho_base=2.0):
    decay = max(0.1, 1.0 - step / total_steps)
    alpha = alpha_base * (1.0 + imb.abs()) * decay
    rho = rho_base * (1.0 + 0.5 * imb.abs()) * decay
    return (alpha * F.softplus(imb) ** rho).mean()


def epistemic_loss(logits, labels, A_prior, root_probs, corr_matrix, model_agreement, step):
    labels = labels.clone()
    labels[labels == 0] = -100
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    A_sample = authority_online(A_prior, model_agreement, sigma_a=CFG.sigma_a)
    H_eff, sh = provenance_online(root_probs, model_agreement, corr_matrix, sigma_h=CFG.sigma_h)
    imb = imbalance(A_sample, H_eff, CFG.sigma_a, sh)
    reg = adaptive_reg(
        imb,
        step,
        total_steps=CFG.total_steps_cap,
        alpha_base=CFG.alpha_base,
        rho_base=CFG.rho_base,
    )
    return ce + CFG.lambda_reg * reg, ce, reg

# =========================================================
# 9. Model Loader
# =========================================================


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(CFG.base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(CFG.device)
    model.train()
    return model, tokenizer


def load_checkpoint_if_available(model) -> Optional[int]:
    ckpt_info = get_last_good_checkpoint()
    if ckpt_info is None:
        return None
    run_id, ckpt_path = ckpt_info
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=CFG.device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Startup] Loaded checkpoint from {ckpt_path} (run {run_id})")
        return run_id
    return None

# =========================================================
# 10. Ingestion + Dedup + Metadata
# =========================================================


def download_file(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return False
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        log_event("WARN", "ingestion", {"msg": "download_failed", "url": url, "error": str(e)})
        return False


def fetch_open_source_docs() -> List[Path]:
    new_files: List[Path] = []
    for archive in CFG.public_archives:
        try:
            resp = requests.get(archive, timeout=20)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            for a in links:
                href = a["href"]
                lower_href = href.lower()
                if not lower_href.endswith((".pdf", ".txt", ".html")):
                    continue
                if not any(kw in lower_href for kw in CFG.keywords):
                    continue
                url = urljoin(archive, href)
                if not url.startswith("http"):
                    continue
                fname = Path(href).name or "downloaded_file"
                out_path = Path(CFG.data_path) / fname
                if out_path.exists():
                    continue
                if not download_file(url, out_path):
                    continue
                text = extract_text(out_path)
                if not text.strip() or DEDUP.is_duplicate(text):
                    out_path.unlink(missing_ok=True)
                    continue
                update_metadata_for_file(out_path, text, source_url=url)
                meta = {
                    "source_url": url,
                    "local_path": str(out_path),
                    "timestamp": time.time(),
                }
                DEDUP.add(text, meta)
                file_hash = compute_text_hash(text)
                record_doc_ingested(out_path, file_hash)
                new_files.append(out_path)
                if len(new_files) >= CFG.max_new_docs_per_cycle:
                    break
        except Exception as e:
            log_event(
                "WARN",
                "ingestion",
                {"msg": "archive_failed", "archive": archive, "error": str(e)},
            )
            continue
    _flush_metadata_buffer()
    return new_files


def continuous_ingestion_loop():
    while True:
        try:
            new_docs = fetch_open_source_docs()
            if new_docs:
                print(f"[Ingestion] Downloaded {len(new_docs)} new (non-duplicate) documents.")
        except Exception as e:
            log_event("ERROR", "ingestion", {"msg": "ingestion_loop_error", "error": str(e)})
        time.sleep(CFG.ingestion_interval_sec)

# =========================================================
# 11. Training Managers (plain + AMP)
# =========================================================


class TrainingManager:
    def __init__(self, model, tokenizer, run_id: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.global_step = 0
        self.training_lock = threading.Lock()
        self.stop_flag = False
        self.run_id = run_id

    def _ensure_run(self):
        if self.run_id is None:
            cfg_dict = {
                k: getattr(CFG, k)
                for k in dir(CFG)
                if not k.startswith("__") and not callable(getattr(CFG, k))
            }
            self.run_id = log_run_start(cfg_dict)

    def _safe_save_checkpoint(self, ckpt_path: Path, state: Dict) -> bool:
        try:
            torch.save(state, ckpt_path)
            return True
        except Exception as e:
            log_event(
                "ERROR",
                "checkpoint",
                {"msg": "save_failed", "path": str(ckpt_path), "error": str(e)},
            )
            return False

    def _save_checkpoint(self, loss: float, ce: float, reg: float):
        self._ensure_run()
        ckpt_path = Path(CFG.checkpoint_dir) / f"run_{self.run_id}_step_{self.global_step}.pt"
        state = {"model_state_dict": self.model.state_dict(), "step": self.global_step}
        if self._safe_save_checkpoint(ckpt_path, state):
            print(f"[Checkpoint] Saved at {ckpt_path}")
            log_run_step(self.run_id, self.global_step, loss, ce, reg, str(ckpt_path))

    def train_cycle(self):
        self._ensure_run()
        dataloader = build_dataloader(self.tokenizer)
        if len(dataloader) == 0:
            return
        total_steps = len(dataloader) * CFG.epochs
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CFG.lr,
            weight_decay=CFG.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(total_steps * CFG.warmup_ratio),
            total_steps,
        )
        try:
            for epoch in range(CFG.epochs):
                for input_ids, attention_mask in dataloader:
                    if self.stop_flag:
                        log_run_end(self.run_id, "interrupted")
                        return
                    with self.training_lock:
                        self.model.train()
                        input_ids = input_ids.to(CFG.device)
                        attention_mask = attention_mask.to(CFG.device)
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        logits = outputs.logits
                        batch_size, seq_len = input_ids.shape
                        _ = sample_source_metadatas(CFG.num_roots)
                        A_prior = torch.rand(batch_size, seq_len, device=CFG.device)
                        root_probs = torch.rand(
                            batch_size, seq_len, CFG.num_roots, device=CFG.device
                        )
                        corr_matrix = (
                            torch.eye(CFG.num_roots, device=CFG.device)
                            .unsqueeze(0)
                            .repeat(batch_size, 1, 1)
                        )
                        model_agreement = torch.rand(batch_size, seq_len, device=CFG.device)
                        loss, ce_loss, reg_loss = epistemic_loss(
                            logits,
                            input_ids,
                            A_prior,
                            root_probs,
                            corr_matrix,
                            model_agreement,
                            self.global_step,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        self.global_step += 1
                        if self.global_step % 50 == 0:
                            print(
                                f"[Train] Epoch {epoch} Step {self.global_step}/{total_steps} "
                                f"Loss={loss.item():.4f} CE={ce_loss.item():.4f} Reg={reg_loss.item():.4f}"
                            )
                        if self.global_step % CFG.checkpoint_every_steps == 0:
                            self._save_checkpoint(loss.item(), ce_loss.item(), reg_loss.item())
            log_run_end(self.run_id, "success")
        except Exception as e:
            log_event("ERROR", "training", {"msg": "train_cycle_error", "error": str(e)})
            if self.run_id is not None:
                log_run_end(self.run_id, "interrupted")

    def training_loop(self):
        while not self.stop_flag:
            self.train_cycle()
            time.sleep(60)


class TrainingManagerAMP:
    def __init__(self, model, tokenizer, run_id: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.global_step = 0
        self.training_lock = threading.Lock()
        self.stop_flag = False
        self.amp_enabled = CFG.device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.run_id = run_id

    def _ensure_run(self):
        if self.run_id is None:
            cfg_dict = {
                k: getattr(CFG, k)
                for k in dir(CFG)
                if not k.startswith("__") and not callable(getattr(CFG, k))
            }
            self.run_id = log_run_start(cfg_dict)

    def _safe_save_checkpoint(self, ckpt_path: Path, state: Dict) -> bool:
        try:
            torch.save(state, ckpt_path)
            return True
        except Exception as e:
            log_event(
                "ERROR",
                "checkpoint",
                {"msg": "save_failed", "path": str(ckpt_path), "error": str(e)},
            )
            return False

    def _save_checkpoint(self, loss: float, ce: float, reg: float):
        self._ensure_run()
        ckpt_path = Path(CFG.checkpoint_dir) / f"run_{self.run_id}_step_{self.global_step}.pt"
        state = {"model_state_dict": self.model.state_dict(), "step": self.global_step}
        if self._safe_save_checkpoint(ckpt_path, state):
            print(f"[Checkpoint] Saved at {ckpt_path}")
            log_run_step(self.run_id, self.global_step, loss, ce, reg, str(ckpt_path))

    def train_streaming_epoch(self):
        self._ensure_run()
        dataloader = build_streaming_dataloader(self.tokenizer)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        total_steps_est = 10_000
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(total_steps_est * CFG.warmup_ratio),
            total_steps_est,
        )
        try:
            for input_ids, attention_mask in dataloader:
                if self.stop_flag:
                    log_run_end(self.run_id, "interrupted")
                    return
                with self.training_lock:
                    self.model.train()
                    input_ids = input_ids.to(
                        CFG.device, non_blocking=(CFG.device == "cuda")
                    )
                    attention_mask = attention_mask.to(
                        CFG.device, non_blocking=(CFG.device == "cuda")
                    )
                    amp_ctx = torch.cuda.amp.autocast(enabled=self.amp_enabled)
                    with amp_ctx:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        logits = outputs.logits
                        batch_size, seq_len = input_ids.shape
                        _ = sample_source_metadatas(CFG.num_roots)
                        A_prior_seq = torch.rand(seq_len, device=CFG.device)
                        root_probs_seq = torch.rand(seq_len, CFG.num_roots, device=CFG.device)
                        corr_matrix_seq = torch.eye(CFG.num_roots, device=CFG.device)
                        A_prior = A_prior_seq.unsqueeze(0).expand(batch_size, -1)
                        root_probs = root_probs_seq.unsqueeze(0).expand(batch_size, -1, -1)
                        corr_matrix = corr_matrix_seq.unsqueeze(0).expand(batch_size, -1, -1)
                        model_agreement = torch.rand(batch_size, seq_len, device=CFG.device)
                        loss, ce_loss, reg_loss = epistemic_loss(
                            logits,
                            input_ids,
                            A_prior,
                            root_probs,
                            corr_matrix,
                            model_agreement,
                            self.global_step,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if self.amp_enabled:
                        try:
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), CFG.max_grad_norm
                            )
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        except RuntimeError as e:
                            log_event(
                                "ERROR",
                                "training_amp",
                                {"msg": "amp_step_failed", "error": str(e)},
                            )
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), CFG.max_grad_norm
                        )
                        optimizer.step()
                    scheduler.step()
                    self.global_step += 1
                    if self.global_step % 50 == 0:
                        print(
                            f"[Train-AMP] Step {self.global_step} "
                            f"Loss={loss.item():.4f} CE={ce_loss.item():.4f} Reg={reg_loss.item():.4f}"
                        )
                    if self.global_step % CFG.checkpoint_every_steps == 0:
                        self._save_checkpoint(loss.item(), ce_loss.item(), reg_loss.item())
            log_run_end(self.run_id, "success")
        except Exception as e:
            log_event(
                "ERROR",
                "training_amp",
                {"msg": "train_streaming_epoch_error", "error": str(e)},
            )
            if self.run_id is not None:
                log_run_end(self.run_id, "interrupted")

    def training_loop(self):
        while not self.stop_flag:
            self.train_streaming_epoch()
            time.sleep(60)

# =========================================================
# 12. Supervisor
# =========================================================


class Supervisor:
    def __init__(self, trainer):
        self.trainer = trainer
        self.stop_flag = False

    def _get_latest_loss(self) -> Optional[float]:
        conn = get_db()
        with _db_lock, conn:
            row = conn.execute(
                "SELECT loss FROM steps ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return row["loss"] if row else None

    def _check_disk_usage(self):
        total, used, free = shutil.disk_usage(".")
        ratio = used / total
        if ratio > CFG.max_disk_usage_ratio:
            log_event("WARN", "supervisor", {"msg": "high_disk_usage", "ratio": ratio})
            ckpts = sorted(
                Path(CFG.checkpoint_dir).glob("*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if len(ckpts) > CFG.max_checkpoints_to_keep:
                to_delete = ckpts[: len(ckpts) - CFG.max_checkpoints_to_keep]
                for p in to_delete:
                    try:
                        p.unlink()
                    except Exception:
                        pass

    def _check_gpu_memory(self):
        if CFG.device == "cuda":
            mem_alloc = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            log_event(
                "INFO",
                "supervisor",
                {
                    "msg": "gpu_memory",
                    "allocated": int(mem_alloc),
                    "reserved": int(mem_reserved),
                },
            )

    def run(self):
        while not self.stop_flag:
            try:
                loss = self._get_latest_loss()
                if loss is not None:
                    print(f"[Supervisor] Latest loss: {loss:.4f}")
                self._check_disk_usage()
                self._check_gpu_memory()
                if CFG.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                log_event(
                    "ERROR",
                    "supervisor",
                    {"msg": "supervisor_error", "error": str(e)},
                )
            time.sleep(CFG.supervisor_interval_sec)

# =========================================================
# 13. Multimodal Encoders + Safe/Cached Wrappers
# =========================================================


class ImageEncoder:
    def __init__(self, device: str):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def encode_path(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return self.encode_pil(img)

    def encode_pil(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self.model.get_image_features(**inputs)  # [1, d]
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().squeeze()


class AudioEncoder:
    def __init__(self, device: str):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperModel.from_pretrained("openai/whisper-small").to(device)
        self.model.eval()

    def encode_path(self, path: Path) -> Tuple[np.ndarray, str]:
        """
        Returns (embedding, transcript_text).
        Embedding is mean-pooled encoder hidden state.
        """
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(self.device)
        inputs = self.processor(
            waveform.squeeze(0),
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        with torch.inference_mode():
            enc_out = self.model.encoder(input_features)
            emb = enc_out.last_hidden_state.mean(dim=1)
            emb = F.normalize(emb, dim=-1).cpu().numpy().squeeze()
        with torch.inference_mode():
            generated_ids = self.model.generate(input_features)
            transcript = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        return emb, transcript


class VideoEncoder:
    def __init__(self, image_encoder: ImageEncoder):
        self.image_encoder = image_encoder

    def encode_path(self, path: Path) -> np.ndarray:
        """
        Sample ~1 frame every ~3 seconds and average CLIP embeddings.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return np.zeros(
                self.image_encoder.model.config.projection_dim, dtype="float32"
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_int = max(int(fps) if fps > 0 else 1, 1)
        frame_interval = max(int(fps_int * 3), 1)  # ~1 frame / 3s
        embeddings: List[np.ndarray] = []
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    emb = self.image_encoder.encode_pil(img)
                    embeddings.append(emb)
                idx += 1
        finally:
            cap.release()
        if embeddings:
            return np.mean(np.stack(embeddings, axis=0), axis=0).astype("float32")
        return np.zeros(self.image_encoder.model.config.projection_dim, dtype="float32")


class MultimodalPrefix(nn.Module):
    """
    Projects modality embeddings into LM hidden space and adds as a prefix bias.
    """

    def __init__(self, model, image_dim: int = 512, audio_dim: int = 384, video_dim: int = 512):
        super().__init__()
        self.model_embed_dim = model.config.hidden_size
        self.img_proj = nn.Linear(image_dim, self.model_embed_dim)
        self.audio_proj = nn.Linear(audio_dim, self.model_embed_dim)
        self.video_proj = nn.Linear(video_dim, self.model_embed_dim)
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.xavier_uniform_(self.audio_proj.weight)
        nn.init.xavier_uniform_(self.video_proj.weight)

    def forward(
        self,
        token_embs: torch.Tensor,
        img_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
        video_emb: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        token_embs: [1, T, H]
        Returns: token_embs with a single fused prefix added to all positions.
        """
        device = token_embs.device
        prefix_vec = torch.zeros(self.model_embed_dim, device=device)
        if img_emb is not None:
            img_t = torch.tensor(img_emb, device=device, dtype=torch.float32)
            prefix_vec = prefix_vec + self.img_proj(img_t)
        if audio_emb is not None:
            audio_t = torch.tensor(audio_emb, device=device, dtype=torch.float32)
            prefix_vec = prefix_vec + self.audio_proj(audio_t)
        if video_emb is not None:
            video_t = torch.tensor(video_emb, device=device, dtype=torch.float32)
            prefix_vec = prefix_vec + self.video_proj(video_t)
        if prefix_vec.abs().sum() == 0:
            return token_embs
        prefix_vec = prefix_vec.unsqueeze(0).unsqueeze(1)  # [1,1,H]
        return token_embs + prefix_vec


def multimodal_condition_to_prefix(embs: List[np.ndarray]) -> str:
    # Now handled in embedding space via MultimodalPrefix; no text marker needed.
    return ""


IMAGE_ENCODER: Optional[ImageEncoder] = None
AUDIO_ENCODER: Optional[AudioEncoder] = None
VIDEO_ENCODER: Optional[VideoEncoder] = None
MM_PREFIX: Optional[MultimodalPrefix] = None


def init_multimodal_components(model):
    global IMAGE_ENCODER, AUDIO_ENCODER, VIDEO_ENCODER, MM_PREFIX
    IMAGE_ENCODER = ImageEncoder(CFG.device)
    AUDIO_ENCODER = AudioEncoder(CFG.device)
    VIDEO_ENCODER = VideoEncoder(IMAGE_ENCODER)
    MM_PREFIX = MultimodalPrefix(model).to(CFG.device)
    MM_PREFIX.eval()

# =========================================================
# 13b. Safe / Cached Media Encoders (bytes-level)
# =========================================================


def safe_encode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    h = sha256_bytes(data)
    cached = load_cached_embedding(h)
    if cached is not None:
        return cached
    try:
        from io import BytesIO
        img = Image.open(BytesIO(data)).convert("RGB")
        emb = IMAGE_ENCODER.encode_pil(img) if IMAGE_ENCODER is not None else None
        if emb is not None:
            save_cached_embedding(h, emb)
        return emb
    except Exception as e:
        log_event("WARN", "image_encoder", {"msg": "encode_failed", "error": str(e)})
        return None


def safe_encode_audio_bytes(data: bytes) -> Tuple[Optional[np.ndarray], Optional[str]]:
    h = sha256_bytes(data)
    cached = load_cached_embedding(h)
    emb: Optional[np.ndarray] = None
    if cached is not None:
        emb = cached
    transcript: Optional[str] = None
    try:
        tmp = media_cache_path(h, ".wav")
        if not tmp.exists():
            tmp.write_bytes(data)
        if emb is None and AUDIO_ENCODER is not None:
            emb, transcript = AUDIO_ENCODER.encode_path(tmp)
            if emb is not None:
                save_cached_embedding(h, emb)
        return emb, transcript
    except Exception as e:
        log_event("WARN", "audio_encoder", {"msg": "encode_failed", "error": str(e)})
        return None, None


def safe_encode_video_bytes(data: bytes) -> Optional[np.ndarray]:
    h = sha256_bytes(data)
    cached = load_cached_embedding(h)
    if cached is not None:
        return cached
    try:
        tmp = media_cache_path(h, ".mp4")
        if not tmp.exists():
            tmp.write_bytes(data)
        emb = VIDEO_ENCODER.encode_path(tmp) if VIDEO_ENCODER is not None else None
        if emb is not None:
            save_cached_embedding(h, emb)
        return emb
    except Exception as e:
        log_event("WARN", "video_encoder", {"msg": "encode_failed", "error": str(e)})
        return None

# =========================================================
# 14. Interactive Chat with Commands (local REPL)
# =========================================================


def print_status():
    conn = get_db()
    with _db_lock, conn:
        run = conn.execute(
            "SELECT * FROM runs ORDER BY start_time DESC LIMIT 1"
        ).fetchone()
        step = conn.execute(
            "SELECT * FROM steps ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        docs_count = conn.execute("SELECT COUNT(*) AS c FROM docs").fetchone()["c"]
    print("=== STATUS ===")
    if run:
        print(
            f"Last run id: {run['id']} status={run['status']} "
            f"last_step={run['last_step']}"
        )
    else:
        print("No runs recorded.")
    if step:
        print(
            f"Last step: {step['step']} loss={step['loss']:.4f} "
            f"ce={step['ce']:.4f} reg={step['reg']:.4f}"
        )
    else:
        print("No steps recorded.")
    print(f"Documents ingested: {docs_count}")


def print_runs(limit: int = 5):
    conn = get_db()
    with _db_lock, conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
    print("=== RUNS ===")
    for r in rows:
        print(
            f"id={r['id']} status={r['status']} last_step={r['last_step']} "
            f"start={time.ctime(r['start_time'])} "
            f"end={time.ctime(r['end_time']) if r['end_time'] else 'N/A'}"
        )


def print_help():
    print("Commands:")
    print(" /status - show last run, last step, docs count")
    print(" /runs   - list recent runs")
    print(" /pause  - pause training")
    print(" /resume - resume training (new thread)")
    print(" /help   - show this help")
    print("Any other text is sent as a prompt to EpsteinGPT.")


def epsteingpt_interact(model, tokenizer, trainer):
    print("EpsteinGPT ready. Type '/help' for commands, 'exit' to quit.")
    while True:
        try:
            line = input(">> ")
        except EOFError:
            break
        cmd = line.strip()
        if cmd.lower() in {"exit", "quit"}:
            trainer.stop_flag = True
            break
        if cmd.startswith("/"):
            if cmd == "/status":
                print_status()
            elif cmd == "/runs":
                print_runs()
            elif cmd == "/pause":
                trainer.stop_flag = True
                print("[REPL] Training pause requested.")
            elif cmd == "/resume":
                if not trainer.stop_flag:
                    print("[REPL] Training already running.")
                else:
                    trainer.stop_flag = False
                    threading.Thread(
                        target=trainer.training_loop, daemon=True
                    ).start()
                    print("[REPL] Training resumed in new thread.")
            elif cmd == "/help":
                print_help()
            else:
                print("[REPL] Unknown command. Use /help.")
            continue

        with trainer.training_lock:
            model.eval()
            with torch.inference_mode():
                inputs = tokenizer(cmd, return_tensors="pt").to(CFG.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(answer)

# =========================================================
# 15. Chat API Schemas + Post-processing
# =========================================================


class ChatImageURL(BaseModel):
    url: str
    detail: Optional[str] = None  # "low" | "high" etc.


class ChatAudioURL(BaseModel):
    url: str


class ChatVideoURL(BaseModel):
    url: str


class ChatMessageContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ChatImageURL] = None
    audio_url: Optional[ChatAudioURL] = None
    video_url: Optional[ChatVideoURL] = None


class ChatMessage(BaseModel):
    role: str
    content: List[ChatMessageContentPart]


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False  # reserved for future streaming


class ChatResponseChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatResponseChoice]


def postprocess_answer(raw: str, simplify: bool = False) -> Dict[str, Any]:
    text = raw.strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    summary_src = paragraphs[0] if paragraphs else text
    summary = summary_src[: CFG.max_summary_chars]

    bullets: List[str] = []
    for p in paragraphs:
        if len(bullets) >= CFG.max_summary_bullets:
            break
        line = p.replace("\n", " ").strip()
        if not line:
            continue
        bullets.append(f"- {line[:240]}")

    out: Dict[str, Any] = {
        "raw": text,
        "summary": summary,
        "bullets": bullets,
    }

    if simplify:
        # Reserved for a second-pass "simplify" call if desired.
        pass

    return out

# =========================================================
# 16. ChatEngine (OpenAI-style) with Multimodal Conditioning
# =========================================================


class ChatEngine:
    """
    OpenAI-style chat engine on top of your causal LM with multimodal conditioning.
    """

    def __init__(self, model, tokenizer, trainer: TrainingManagerAMP):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.lock = threading.Lock()

    def _get_session(self, session_id: str) -> List[ChatMessage]:
        return self.sessions.setdefault(session_id, [])

    def _collect_modal_embeddings(
        self, msg: ChatMessage
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        img_emb = None
        audio_emb = None
        video_emb = None
        debug = {
            "image_encoded": False,
            "audio_encoded": False,
            "video_encoded": False,
        }

        for part in msg.content:
            if part.type == "image_url" and part.image_url and IMAGE_ENCODER is not None:
                url = part.image_url.url
                try:
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    if len(resp.content) > CFG.max_upload_mb * 1024 * 1024:
                        log_event(
                            "WARN",
                            "image_encoder",
                            {"msg": "file_too_large", "url": url},
                        )
                        continue
                    emb = safe_encode_image_bytes(resp.content)
                    if emb is not None:
                        img_emb = emb
                        debug["image_encoded"] = True
                except Exception as e:
                    log_event(
                        "WARN",
                        "image_encoder",
                        {"msg": "download_failed", "url": url, "error": str(e)},
                    )

            elif part.type == "audio_url" and part.audio_url and AUDIO_ENCODER is not None:
                url = part.audio_url.url
                try:
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    if len(resp.content) > CFG.max_upload_mb * 1024 * 1024:
                        log_event(
                            "WARN",
                            "audio_encoder",
                            {"msg": "file_too_large", "url": url},
                        )
                        continue
                    emb, _ = safe_encode_audio_bytes(resp.content)
                    if emb is not None:
                        audio_emb = emb
                        debug["audio_encoded"] = True
                except Exception as e:
                    log_event(
                        "WARN",
                        "audio_encoder",
                        {"msg": "download_failed", "url": url, "error": str(e)},
                    )

            elif part.type == "video_url" and part.video_url and VIDEO_ENCODER is not None:
                url = part.video_url.url
                try:
                    resp = requests.get(url, timeout=120)
                    resp.raise_for_status()
                    if len(resp.content) > CFG.max_upload_mb * 1024 * 1024:
                        log_event(
                            "WARN",
                            "video_encoder",
                            {"msg": "file_too_large", "url": url},
                        )
                        continue
                    emb = safe_encode_video_bytes(resp.content)
                    if emb is not None:
                        video_emb = emb
                        debug["video_encoded"] = True
                except Exception as e:
                    log_event(
                        "WARN",
                        "video_encoder",
                        {"msg": "download_failed", "url": url, "error": str(e)},
                    )

        return img_emb, audio_emb, video_emb, debug

    def _flatten_message_text(self, msg: ChatMessage) -> str:
        parts = [p.text for p in msg.content if p.type == "text" and p.text]
        return "\n".join(parts).strip()

    def _build_prompt(self, history: List[ChatMessage], newmsg: ChatMessage) -> str:
        lines: List[str] = []
        for m in history:
            txt = self._flatten_message_text(m)
            if txt:
                lines.append(f"{m.role.upper()}: {txt}")
        newtxt = self._flatten_message_text(newmsg)
        if newtxt:
            lines.append(f"{newmsg.role.upper()}: {newtxt}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _extract_assistant_reply(self, full_text: str) -> str:
        segments = full_text.split("ASSISTANT:")
        if len(segments) == 0:
            return full_text.strip()
        return segments[-1].strip()

    def chat(self, req: ChatRequest, session_id: str = "default") -> ChatResponse:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        history_messages = req.messages[:-1]
        newmsg = req.messages[-1]
        if newmsg.role not in ("user", "system"):
            raise HTTPException(
                status_code=400,
                detail="Last message must be user or system",
            )

        session_history = self._get_session(session_id)
        if history_messages:
            session_history.extend(history_messages)

        prompt = self._build_prompt(session_history, newmsg)
        img_emb, audio_emb, video_emb, debug = self._collect_modal_embeddings(newmsg)

        with self.trainer.training_lock:
            self.model.eval()
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(CFG.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                if (
                    MM_PREFIX is not None
                    and (img_emb is not None or audio_emb is not None or video_emb is not None)
                ):
                    tok_embs = self.model.transformer.wte(input_ids)
                    fused_embs = MM_PREFIX(
                        tok_embs, img_emb=img_emb, audio_emb=audio_emb, video_emb=video_emb
                    )
                    outputs = self.model.generate(
                        inputs_embeds=fused_embs,
                        attention_mask=attention_mask,
                        max_new_tokens=req.max_tokens,
                        do_sample=True,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=req.max_tokens,
                        do_sample=True,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                reply_text = self._extract_assistant_reply(full_text)

        processed = postprocess_answer(reply_text)

        content_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": reply_text},
        ]
        if processed["summary"]:
            content_parts.append(
                {"type": "text", "text": "Summary:\n" + processed["summary"]}
            )
        if processed["bullets"]:
            content_parts.append(
                {
                    "type": "text",
                    "text": "Key points:\n" + "\n".join(processed["bullets"]),
                }
            )

        if (
            img_emb is None
            and audio_emb is None
            and video_emb is None
            and any(p.type != "text" for p in newmsg.content)
        ):
            content_parts.append(
                {
                    "type": "text",
                    "text": (
                        "Note: Some attached files could not be processed. "
                        "Try a different format or a shorter clip."
                    ),
                }
            )

        session_history.append(newmsg)
        assistant_msg = ChatMessage(
            role="assistant",
            content=[ChatMessageContentPart(type="text", text=reply_text)],
        )
        session_history.append(assistant_msg)

        choice = ChatResponseChoice(
            index=0,
            message={
                "role": "assistant",
                "content": content_parts,
                "debug": debug,
            },
            finish_reason="stop",
        )

        resp = ChatResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model or CFG.base_model,
            choices=[choice],
        )
        return resp

# =========================================================
# 17. FastAPI App + Initialization
# =========================================================

app = FastAPI(title="EpsteinGPT API")
chat_engine: Optional[ChatEngine] = None


@app.post("/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest = Body(...), session_id: str = "default"):
    global chat_engine
    if chat_engine is None:
        raise HTTPException(status_code=500, detail="ChatEngine not initialized")
    return chat_engine.chat(req, session_id=session_id)

# =========================================================
# 18. Main Entrypoint
# =========================================================


def main():
    init_db()
    load_metadata_store()

    # Start ingestion thread
    threading.Thread(target=continuous_ingestion_loop, daemon=True).start()

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer()
    existing_run_id = load_checkpoint_if_available(model)

    # Training manager with AMP
    trainer = TrainingManagerAMP(model, tokenizer, run_id=existing_run_id)
    supervisor = Supervisor(trainer)

    # Start training and supervisor threads
    threading.Thread(target=trainer.training_loop, daemon=True).start()
    threading.Thread(target=supervisor.run, daemon=True).start()

    # Initialize multimodal components
    init_multimodal_components(model)

    # Initialize ChatEngine for API
    global chat_engine
    chat_engine = ChatEngine(model, tokenizer, trainer)

    # Run API server in background if enabled
    if CFG.enable_api:
        def run_api():
            uvicorn.run(app, host=CFG.api_host, port=CFG.api_port, log_level="info")

        threading.Thread(target=run_api, daemon=True).start()

    # Local REPL remains available
    try:
        epsteingpt_interact(model, tokenizer, trainer)
    finally:
        DEDUP.close()
        _flush_metadata_buffer()


if __name__ == "__main__":
    main()
