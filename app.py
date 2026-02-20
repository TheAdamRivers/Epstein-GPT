import json
import time
from typing import List, Dict, Any

import requests
import streamlit as st


# ---------- Core config ----------

API_URL = "http://localhost:8000/chat/completions"
SESSION_ID = "default"
REQUEST_TIMEOUT = 300  # seconds
MAX_HISTORY_MESSAGES = 10  # keep last N turns for latency


# ---------- Page config & global styling ----------

st.set_page_config(
    page_title="EpsteinGPT – SynthicSoft Labs",
    page_icon="🧠",
    layout="wide",
)

# Global theming / layout tweaks
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 55%);
        color: #e5e7eb;
    }
    /* Hide default padding at the top */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
        max-width: 1100px;
    }
    .synthic-header {
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 0.5rem;
    }
    .synthic-header h1 {
        font-size: 1.8rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .synthic-header h1 span.logo {
        font-size: 1.9rem;
    }
    .synthic-header p {
        font-size: 0.9rem;
        color: #9ca3af;
        margin: 0;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.75rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: #9ca3af;
        margin-right: 0.35rem;
    }
    .synthic-footer {
        margin-top: 1.5rem;
        font-size: 0.8rem;
        color: #6b7280;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
        padding-top: 0.6rem;
    }
    .synthic-footer a {
        color: #93c5fd;
        text-decoration: none;
    }
    .synthic-footer a:hover {
        text-decoration: underline;
    }
    /* Chat tweaks */
    .stChatMessage {
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .st-emotion-cache-1r6slb0 {  /* chat input container */
        border-top: 1px solid rgba(148, 163, 184, 0.35);
        padding-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="synthic-header">
      <h1><span class="logo">🧠</span> EpsteinGPT</h1>
      <p>by SynthicSoft Labs · Local-first epistemic assistant over public archives</p>
      <div style="margin-top:0.3rem;">
        <span class="pill">alpha · research build</span>
        <span class="pill">offline-friendly</span>
        <span class="pill">open source</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Sidebar: branding, persona, and controls ----------

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 256
if "persona" not in st.session_state:
    st.session_state.persona = "Epistemic analyst"

with st.sidebar:
    st.markdown("### SynthicSoft Labs")
    st.markdown(
        "Local‑first, security‑focused AI tooling, built for practitioners.\n\n"
        "[🌐 Website](https://synthicsoftlabs.com)"
    )

    st.markdown("---")
    st.markdown("#### Assistant style")

    persona = st.selectbox(
        "Persona",
        [
            "Epistemic analyst",
            "Dry researcher",
            "Explainer for non‑technical users",
            "Custom",
        ],
        index=["Epistemic analyst", "Dry researcher",
               "Explainer for non‑technical users", "Custom"].index(st.session_state.persona)
        if st.session_state.persona in [
            "Epistemic analyst",
            "Dry researcher",
            "Explainer for non‑technical users",
            "Custom",
        ]
        else 0,
    )
    st.session_state.persona = persona

    custom_system_prompt = ""
    if persona == "Epistemic analyst":
        system_prompt = (
            "You are EpsteinGPT, an epistemic analyst over public archives. "
            "You reason step by step, call out uncertainty, and avoid listing specific "
            "real-world individuals as criminals unless quoting well-documented public records."
        )
    elif persona == "Dry researcher":
        system_prompt = (
            "You are EpsteinGPT, a dry, technical research assistant. "
            "You answer concisely in neutral tone, always noting limitations and uncertainty."
        )
    elif persona == "Explainer for non‑technical users":
        system_prompt = (
            "You are EpsteinGPT, an explainer for non‑technical users. "
            "You use plain language, short paragraphs, and concrete examples when helpful."
        )
    else:  # Custom
        custom_system_prompt = st.text_area(
            "Custom system prompt",
            value=st.session_state.get(
                "custom_system_prompt_value",
                "You are EpsteinGPT, a helpful local research assistant.",
            ),
            height=120,
        )
        st.session_state.custom_system_prompt_value = custom_system_prompt
        system_prompt = custom_system_prompt or (
            "You are EpsteinGPT, a helpful local research assistant."
        )

    st.markdown("---")
    st.markdown("#### Generation")

    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05
    )
    st.session_state.max_tokens = st.slider(
        "Max response tokens", 32, 512, st.session_state.max_tokens, 16
    )

    fast_mode = st.checkbox("Fast mode (shorter answers)", value=True)
    st.session_state.fast_mode = fast_mode

    st.markdown("#### Session")
    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.experimental_rerun()


# ---------- Backend helpers ----------

def build_backend_messages(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert Streamlit chat history into the backend ChatMessage format:
    {
      "role": "user" | "assistant" | "system",
      "content": [{"type": "text", "text": "..."}]
    }
    """
    backend_messages: List[Dict[str, Any]] = []

    # System message from selected persona
    backend_messages.append(
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
    )

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
    """
    Sends (truncated) history to the backend and returns cleaned assistant text.

    - Limits history length for latency.
    - Prefers main raw answer and optional 'Summary:' line.
    - Skips 'Key points:' blocks.
    - Adds simple safety/quality guards.
    """
    short_history = history[-MAX_HISTORY_MESSAGES:]
    messages = build_backend_messages(short_history)

    # Backend requires last message role to be user or system
    if not messages or messages[-1]["role"] not in ("user", "system"):
        raise ValueError("Last message must be user or system for backend request.")

    max_tokens = st.session_state.get("max_tokens", 256)
    if st.session_state.get("fast_mode", False):
        max_tokens = min(max_tokens, 160)

    payload: Dict[str, Any] = {
        "model": "epsteingpt",
        "messages": messages,
        "temperature": st.session_state.get("temperature", 0.7),
        "top_p": 0.9,
        "max_tokens": max_tokens,
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

    message = choices[0].get("message", {})
    parts = message.get("content", [])

    raw = None
    summary = None

    # Extract main answer and optional summary; skip key points and noise
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue

        text = (part.get("text") or "").strip()
        if not text:
            continue

        if text.startswith("Summary:"):
            summary = text
        elif text.startswith("Key points:"):
            # Skip key points in the main chat display for now
            continue
        else:
            # First non-summary, non-key-points text is treated as main answer
            if raw is None:
                raw = text

    # Fallback: if we didn't find a clear raw answer, join all text parts
    if raw is None:
        texts: List[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                t = (part.get("text") or "").strip()
                if t:
                    texts.append(t)
        if not texts:
            return json.dumps(message, ensure_ascii=False)
        raw = "\n\n".join(texts)

    raw = raw.strip()

    # --- Basic quality filters for obviously bad answers ---

    user_last = history[-1]["content"].strip().lower() if history else ""

    # If the user asked "who ..." and the answer is tiny (e.g. "yes.")
    if user_last.startswith("who ") and len(raw) < 20:
        return (
            "I’m not able to provide a reliable list of specific names from the Epstein "
            "files in this local model snapshot. You should consult primary sources and "
            "verified reporting for that question."
        )

    # If the answer looks like a random transcript (speaker tags in all caps + colon)
    if any(token.endswith(":") and token.isupper() for token in raw.split()):
        return (
            "The model drifted into unrelated transcript-like text instead of answering "
            "your question clearly. Please rephrase or narrow your question."
        )

    if summary:
        return raw + "\n\n\n" + summary.strip()

    return raw


# ---------- Chat UI ----------

if "messages" not in st.session_state:
    # Each message: {"role": "user" | "assistant" | "system", "content": str}
    st.session_state.messages: List[Dict[str, str]] = []
    st.markdown(
        "#### Welcome\n"
        "_EpsteinGPT is an experimental, local‑first research assistant. Responses may be slow and should be independently verified._"
    )

# Render existing history
for msg in st.session_state.messages:
    role = msg["role"]
    if role not in ("user", "assistant"):
        role = "assistant"
    avatar = "🧑‍💻" if role == "user" else "🧠"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# Input from user
user_text = st.chat_input("Ask EpsteinGPT anything about the public archives…")

if user_text:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Display user message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_text)

    # Backend call + assistant rendering
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("EpsteinGPT is thinking…"):
            start = time.time()
            try:
                assistant_text = call_backend(st.session_state.messages)
                elapsed = time.time() - start

                # Streaming façade for better UX
                placeholder = st.empty()
                buffer = ""
                words = assistant_text.split(" ")
                # Speed up if answer is long
                delay = 0.02 if len(words) < 120 else 0.0

                for chunk in words:
                    buffer += chunk + " "
                    placeholder.markdown(buffer + "▌")
                    if delay > 0:
                        time.sleep(delay)
                placeholder.markdown(buffer)

                # Persist in history
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )

                # Latency + mode info
                mode_label = "fast" if st.session_state.get("fast_mode", False) else "deep"
                st.caption(
                    f"Response time: {elapsed:.2f} s · mode: {mode_label} · "
                    f"T={st.session_state.temperature:.2f}, max_tokens={st.session_state.max_tokens}"
                )

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


# ---------- Footer ----------

st.markdown(
    """
    <div class="synthic-footer">
      EpsteinGPT is an experimental open-source research system from SynthicSoft Labs.
      Learn more at <a href="https://synthicsoftlabs.com" target="_blank">synthicsoftlabs.com</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
