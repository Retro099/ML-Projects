import streamlit as st
from transformers import pipeline
from pathlib import Path

st.set_page_config(page_title="Japanese Sentiment", page_icon="🇯🇵", layout="centered")
st.title("🇯🇵 Japanese Sentiment Analysis")
st.caption("Day 3 Production Model – Ready for Streamlit Cloud")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cl-tohoku/bert-base-japanese-v2", 
                    device=0 if st.runtime.exists("cuda") else -1)

classifier = load_model()

text = st.text_area("レビューを入力してください", height=150)
if st.button("分析する", type="primary"):
    with st.spinner("分析中..."):
        result = classifier(text)[0]
        label_map = {"positive": "Positive 😊", "negative": "Negative 😞", "neutral": "Neutral 😐"}
        st.success(f"**予測:** {label_map.get(result['label'].lower(), result['label'])}")
        st.info(f"Confidence: {result['score']:.1%}")
