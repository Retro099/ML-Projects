import streamlit as st
from transformers import pipeline

st.title("🇯🇵 Japanese Sentiment Analysis")
st.write("Day 2 baseline model (Positive / Neutral / Negative)")

classifier = pipeline("sentiment-analysis", 
                      model="cl-tohoku/bert-base-japanese-v2", 
                      device=0 if st.runtime.exists("cuda") else -1)

text = st.text_area("レビューを入力してください", "この商品は最高です！とても満足しています。")
if st.button("分析する"):
    result = classifier(text)[0]
    label_map = {"positive": "Positive 😊", "negative": "Negative 😞", "neutral": "Neutral 😐"}
    st.success(f"予測: {label_map.get(result['label'].lower(), result['label'])} (score: {result['score']:.3f})")
