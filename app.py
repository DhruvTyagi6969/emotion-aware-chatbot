import streamlit as st
import torch
import time
import os
import random
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

# ───────────────────────── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(page_title="Emotion Chatbot", layout="centered", page_icon="💚")

# ───────────────────────── TERMINAL GREEN THEME ────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Header ── */
h1 {
    color: #3fb950;
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    text-shadow: 0 0 20px rgba(63,185,80,0.3);
}

/* ── Subtitle ── */
.subtitle {
    text-align: center;
    color: #484f58;
    font-size: 13px;
    margin-top: -10px;
    margin-bottom: 20px;
}

/* ── Input box ── */
.stTextInput input {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    padding: 13px 18px !important;
    border: 1px solid #30363d !important;
    font-size: 14.5px !important;
    font-family: 'Segoe UI', monospace !important;
    box-shadow: none !important;
    transition: border-color 0.2s;
}
.stTextInput input:focus {
    border-color: #3fb950 !important;
    box-shadow: 0 0 0 3px rgba(63,185,80,0.12) !important;
}
.stTextInput input::placeholder { color: #484f58 !important; }
.stTextInput label {
    color: #8b949e !important;
    font-size: 13px !important;
}

/* ── User bubble ── */
.user-bubble {
    background: #238636;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 6px 6px 2px 6px;
    margin: 6px 0 6px auto;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #2ea043;
    box-shadow: 0 1px 4px rgba(35,134,54,0.25);
}

/* ── Bot bubble ── */
.bot-bubble {
    background: #161b22;
    color: #c9d1d9;
    padding: 12px 16px;
    border-radius: 6px 6px 6px 2px;
    margin: 6px 0;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #30363d;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}

/* ── Emotion badges ── */
.emotion {
    margin-top: 8px;
    font-size: 10.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    display: inline-block;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: monospace;
}
.emotion-anger    { background: #3d0000; color: #ff7b72; border: 1px solid #ff7b72; }
.emotion-disgust  { background: #1e003d; color: #d2a8ff; border: 1px solid #d2a8ff; }
.emotion-fear     { background: #3d1500; color: #ffa657; border: 1px solid #ffa657; }
.emotion-joy      { background: #003820; color: #3fb950; border: 1px solid #3fb950; }
.emotion-love     { background: #3d0020; color: #f778ba; border: 1px solid #f778ba; }
.emotion-neutral  { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }
.emotion-sadness  { background: #001f3d; color: #79c0ff; border: 1px solid #79c0ff; }
.emotion-surprise { background: #3d3000; color: #e3b341; border: 1px solid #e3b341; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] h3 {
    color: #3fb950 !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background-color: #3fb950 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    border: 1px solid #30363d;
    color: #8b949e;
    border-radius: 6px;
    font-size: 13px;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #3fb950;
    color: #3fb950;
}

/* ── Welcome card ── */
.welcome-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 26px;
    text-align: center;
    margin: 8px 0 24px 0;
    border-top: 3px solid #3fb950;
}
.welcome-card h3 {
    color: #3fb950;
    font-size: 1.1rem;
    margin-bottom: 10px;
    font-family: monospace;
}
.welcome-card p {
    color: #8b949e;
    font-size: 13.5px;
    line-height: 1.65;
    font-family: monospace;
}

/* ── Divider ── */
hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("<h1>💚 Emotion Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>phase-2 · roberta-finetuned · gemini-2.5-flash · 8 emotions</p>",
    unsafe_allow_html=True
)

# ───────────────────────── LABEL CONFIG ────────────────────────────────────────
# All 28 GoEmotions labels in training index order
EMOTION_LABELS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"
]

# Collapse 28 → 8 for display and logic
SIMPLIFIED_MAP = {
    "admiration": "joy",     "amusement": "joy",       "anger": "anger",
    "annoyance": "anger",    "approval": "joy",         "caring": "love",
    "confusion": "neutral",  "curiosity": "neutral",    "desire": "love",
    "disappointment": "sadness", "disapproval": "anger", "disgust": "disgust",
    "embarrassment": "fear", "excitement": "joy",       "fear": "fear",
    "gratitude": "joy",      "grief": "sadness",        "joy": "joy",
    "love": "love",          "nervousness": "fear",     "optimism": "joy",
    "pride": "joy",          "realization": "neutral",  "relief": "joy",
    "remorse": "sadness",    "sadness": "sadness",      "surprise": "surprise",
    "neutral": "neutral"
}

BADGE_ICONS = {
    "anger": "🔴", "disgust": "🟣", "fear": "🟠",
    "joy": "🟢",   "love": "🩷",   "neutral": "⚪",
    "sadness": "🔵", "surprise": "🟡"
}

NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "disgust"}

# ───────────────────────── LOAD MODELS ─────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    '''Load YOUR fine-tuned RoBERTa from emotion_model_v2'''
    path = "./emotion_model_v2"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("models/gemini-2.5-flash")

emotion_tokenizer, emotion_model = load_emotion_model()
llm = load_llm()

# ───────────────────────── HELPER: SAFE HTML ───────────────────────────────────
def safe_html(text):
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )

# ───────────────────────── EMOTION PREDICTION ──────────────────────────────────
def predict_emotion(text):
    '''Hybrid: keyword shortcuts → fine-tuned model fallback'''
    t = text.lower()

    # Keyword shortcuts (fast path)
    keyword_map = [
        (["very sad", "crying", "devastated", "heartbroken"], "sadness"),
        (["so happy", "amazing", "wonderful", "great day"],   "joy"),
        (["furious", "very angry", "rage"],                   "anger"),
        (["terrified", "scared", "afraid"],                   "fear"),
        (["i love", "in love"],                               "love"),
        (["disgusting", "gross", "revolting"],                "disgust"),
        (["wow", "unexpected", "surprised"],                  "surprise"),
        (["fine", "ok", "okay", "all good", "alright"],       "neutral"),
    ]
    for phrases, emotion in keyword_map:
        if any(p in t for p in phrases):
            return [(emotion, 0.95)]

    # Fine-tuned model
    inputs = emotion_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=64
    )
    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    probs    = torch.sigmoid(logits)[0]
    top_idx  = torch.argmax(probs).item()
    raw_label = EMOTION_LABELS_28[top_idx]
    simplified = SIMPLIFIED_MAP.get(raw_label, "neutral")

    return [(simplified, round(probs[top_idx].item(), 3))]

# ───────────────────────── SAFETY CHECK ────────────────────────────────────────
def safety_check(text):
    '''Combines hard word list + model-detected anger/disgust'''
    hard_block = ["fuck", "shit", "bitch", "asshole", "cunt", "bastard"]
    if any(w in text.lower() for w in hard_block):
        return True

    # Also block if model detects strong anger or disgust
    emotions = predict_emotion(text)
    if emotions[0][0] in {"anger", "disgust"} and emotions[0][1] > 0.75:
        return True

    return False

# ───────────────────────── SMART MEMORY ────────────────────────────────────────
def get_memory_summary():
    '''Detects repeated negative patterns across the conversation'''
    history  = st.session_state.chat
    bot_msgs = [m for m in history if m[0] == "Bot" and len(m) > 2]

    if not bot_msgs:
        return "No emotional context yet."

    recent_emotions = [m[2] for m in bot_msgs[-5:]]
    counts = Counter(recent_emotions)
    dominant = counts.most_common(1)[0]

    # Repeated negative → add crisis-aware note
    if dominant[0] in NEGATIVE_EMOTIONS and dominant[1] >= 3:
        return (
            f"User has shown {dominant[0]} in {dominant[1]} of the last "
            f"{len(recent_emotions)} messages. Show extra care and gently "
            f"suggest professional support if appropriate."
        )

    return f"User most recently expressed {recent_emotions[-1]}."

# ───────────────────────── RESPONSE GENERATION ─────────────────────────────────
FALLBACK_RESPONSES = [
    "I'm here with you. Do you want to share more?",
    "That sounds important. I'm listening.",
    "Take your time, I'm here for you.",
    "I hear you. Want to talk about it more?",
]

def generate_response(emotions, user_input):
    emotion = emotions[0][0]
    st.session_state.last_emotion = emotion

    # API limit guard
    if st.session_state.get("api_calls", 0) >= 50:
        return random.choice(FALLBACK_RESPONSES)

    history_str = ""
    for msg in st.session_state.chat[-6:]:
        role = "User" if msg[0] == "You" else "Bot"
        history_str += f"{role}: {msg[1]}\n"

    prompt = f"""
You are an emotionally intelligent mental health chatbot.

Detected Emotion : {emotion}
Session Memory   : {get_memory_summary()}

RULES:
- Reply in 2–4 sentences max
- Use the user's own words where natural
- Be warm, not robotic
- If memory shows repeated distress, gently acknowledge and suggest support
- Never diagnose

Conversation so far:
{history_str}
User: {user_input}
Bot:"""

    try:
        res = llm.generate_content(prompt)
        st.session_state.api_calls = st.session_state.get("api_calls", 0) + 1
        if hasattr(res, "text") and res.text:
            return res.text.strip()
        return "I'm here, but had trouble responding."
    except Exception as e:
        if "429" in str(e):
            return random.choice(FALLBACK_RESPONSES)
        return f"Error: {str(e)}"

# ───────────────────────── SESSION INIT ────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = []
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

# ───────────────────────── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💚 Emotion Chatbot")
    st.caption("phase-2 · roberta + gemini")
    st.markdown("---")

    st.markdown("### 📊 Session Emotions")
    bot_emotions = [m[2] for m in st.session_state.chat if m[0] == "Bot" and len(m) > 2]
    if bot_emotions:
        counts = Counter(bot_emotions)
        for emo, cnt in counts.most_common():
            icon = BADGE_ICONS.get(emo, "⚪")
            pct = int((cnt / len(bot_emotions)) * 100)
            st.markdown(f"{icon} **{emo}** — {cnt} msg ({pct}%)")
    else:
        st.caption("No messages yet.")

    st.markdown("---")
    used = st.session_state.get("api_calls", 0)
    st.caption(f"API calls: {used} / 50")
    st.progress(used / 50)

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.chat = []
        st.session_state.api_calls = 0
        st.rerun()

# ───────────────────────── WELCOME CARD ────────────────────────────────────────
if not st.session_state.chat:
    st.markdown("""
    <div class='welcome-card'>
        <h3>$ emotion_chatbot --start</h3>
        <p>> initialising RoBERTa model...<br>
        > connecting to Gemini API...<br>
        > ready. tell me how you feel.<br><br>
        detects: joy · sadness · anger · fear · love · disgust · surprise · neutral</p>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────── INPUT HANDLER ───────────────────────────────────────
def handle_submit():
    text = st.session_state.user_input.strip()
    if not text:
        return

    if safety_check(text):
        st.session_state.chat.append(("You", text))
        st.session_state.chat.append(("Bot", "Let's keep things respectful. I'm still here for you.", "neutral"))
    else:
        emo   = predict_emotion(text)
        reply = generate_response(emo, text)
        st.session_state.chat.append(("You", text))
        st.session_state.chat.append(("Bot", reply, emo[0][0]))

    st.session_state.user_input = ""

st.text_input("$ input:", key="user_input", on_change=handle_submit, placeholder="type how you feel...")

# ───────────────────────── BADGE HELPER ────────────────────────────────────────
def get_badge(emotion):
    icon = BADGE_ICONS.get(emotion, "⚪")
    return f"<div class='emotion emotion-{emotion}'>{icon} {emotion}</div>"

# ───────────────────────── CHAT DISPLAY ────────────────────────────────────────
for i, item in enumerate(st.session_state.chat):
    if item[0] == "You":
        st.markdown(
            f"<div class='user-bubble'>🧑 {safe_html(item[1])}</div>",
            unsafe_allow_html=True
        )
    else:
        emotion   = item[2]
        safe_text = safe_html(item[1])
        is_last   = (i == len(st.session_state.chat) - 1)

        if is_last:
            placeholder = st.empty()
            typed = ""
            for word in item[1].split():
                typed += word + " "
                placeholder.markdown(
                    f"<div class='bot-bubble'>🤖 {safe_html(typed)}▌</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.03)
            placeholder.markdown(
                f"<div class='bot-bubble'>🤖 {safe_text}<br>{get_badge(emotion)}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='bot-bubble'>🤖 {safe_text}<br>{get_badge(emotion)}</div>",
                unsafe_allow_html=True
            )