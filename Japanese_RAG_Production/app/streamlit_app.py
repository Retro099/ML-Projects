"""Streamlit frontend for Japanese RAG System."""

import streamlit as st
import requests
import time
import html as html_lib
from typing import Dict, Any


API_URL = "http://127.0.0.1:8000/ask"


def ask_rag(question: str, top_k: int) -> Dict[str, Any]:
    """Send a question to the FastAPI backend and return the response."""
    response = requests.post(
        API_URL,
        json={"question": question, "top_k": top_k},
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def main():
    st.set_page_config(page_title="JP Japanese RAG System", layout="wide")
    st.title("JP Japanese RAG System")
    st.subheader("Production-Ready Retrieval Augmented Generation Demo")

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of chunks to retrieve", 3, 10, 5)

    st.markdown("### Ask a Question")
    query = st.text_input("Enter your question in Japanese or English:")

    if st.button("Get Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking..."):
            start_time = time.time()

            try:
                data = ask_rag(query, top_k)
                total_time = time.time() - start_time

                answer = data.get("answer", "No answer generated.")
                sources = data.get("sources", [])
                chunks_count = data.get("chunks_retrieved", 0)

                st.markdown("### Answer")

                # Highlighted answer box
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1E1E1E;
                        border: 1px solid #333333;
                        border-left: 5px solid #FF4B4B;
                        border-radius: 8px;
                        padding: 16px 20px;
                        margin-top: 8px;
                        margin-bottom: 20px;
                        font-size: 1.05rem;
                        line-height: 1.6;
                        white-space: pre-line;
                    ">
                        {html_lib.escape(answer)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)
                col1.metric("Chunks Retrieved", chunks_count)
                col2.metric("Total Response Time", f"{total_time:.2f}s")

                with st.expander(f"Sources Used ({len(sources)} chunks)"):
                    for i, source in enumerate(sources, 1):
                        filename = source.get("metadata", {}).get("filename", "Unknown")
                        content = source.get("content", "")
                        st.markdown(f"**Source {i}** — `{filename}`")
                        st.text(content[:500] + "..." if len(content) > 500 else content)
                        st.divider()

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to FastAPI backend: {e}")
                st.info("Make sure your FastAPI server is running on http://127.0.0.1:8000")


if __name__ == "__main__":
    main()