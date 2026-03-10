import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Japanese Sentiment", page_icon="🇯🇵", layout="centered")
st.title("🇯🇵 Japanese Sentiment Analysis")
st.caption("Day 4 Production Model – Deployed on Streamlit Cloud")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return pipeline("sentiment-analysis", 
                    model="Retro099/japanese-sentiment-analysis-v1",
                    device=-1)   # CPU only

classifier = load_model()

# Professional label mapping
label_map = {
    "LABEL_0": "Negative 😞",
    "LABEL_1": "Neutral 😐",
    "LABEL_2": "Positive 😊"
}

text = st.text_area("レビューを入力してください", height=150)
if st.button("分析する", type="primary"):
    with st.spinner("分析中..."):
        result = classifier(text)[0]
        prediction = label_map.get(result['label'], result['label'])
        st.success(f"**予測:** {prediction}")
        st.info(f"Confidence: {result['score']:.1%}")