"""
app.py — Streamlit web interface for the spam detector.
Run with: streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .result-spam {
        background: #ffe5e5; border-left: 5px solid #e53e3e;
        padding: 1rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-ham {
        background: #e5ffe8; border-left: 5px solid #38a169;
        padding: 1rem; border-radius: 8px; margin-top: 1rem;
    }
    .confidence-bar { height: 12px; border-radius: 6px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ AI Spam Detector")
st.markdown("Paste any email or SMS message below to check if it's **spam or legitimate**.")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
message = st.text_area(
    "✉️ Enter your message:",
    height=150,
    placeholder="e.g. Congratulations! You've won a FREE gift card. Click here to claim now!",
)

# ── Example buttons ───────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚨 Spam example"):
        message = "FREE entry in 2 a weekly comp to win FA Cup final tkts! Text FA to 87121 to receive entry question(std txt rate)"
with col2:
    if st.button("✅ Ham example"):
        message = "Hey! Are you free this weekend? We're having a small get-together at my place."

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Analyse Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analysing..."):
            result = predict(message)

        label      = result["label"]
        confidence = result["confidence"]
        pct        = int(confidence * 100)
        css_class  = "result-spam" if label == "Spam" else "result-ham"
        icon       = "🚨" if label == "Spam" else "✅"
        color      = "#e53e3e" if label == "Spam" else "#38a169"

        st.markdown(f"""
        <div class="{css_class}">
            <h3>{icon} {label}</h3>
            <p>Confidence: <strong>{pct}%</strong></p>
            <div style="background:#ddd;" class="confidence-bar">
                <div class="confidence-bar"
                     style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Extra advice
        if label == "Spam":
            st.error("⚠️ Do not click any links in this message. It shows common spam patterns.")
        else:
            st.success("This message looks legitimate.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Naive Bayes + TF-IDF  |  Dataset: UCI SMS Spam Collection  |  Accuracy: ~97%")
