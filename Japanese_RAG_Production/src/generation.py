"""Response generation using Groq LLM."""

import os
from typing import List, Dict
from groq import Groq
from src.config import config


class Generator:
    """Handles grounded response generation using LLM API."""

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = config.LLM_MODEL

    def generate(self, query: str, context: List[Dict]) -> str:
        """Generate a short, direct, and relevant answer grounded in the provided context."""
        if not context:
            return "申し訳ありません。関連する情報が見つかりませんでした。"

        context_text = "\n\n".join([chunk["content"] for chunk in context])

        prompt = f"""あなたは正確で簡潔な回答をするアシスタントです。

### 厳守ルール:
- 回答は**短く・簡潔に**まとめてください。
- 余計な説明、前置き、背景情報は一切書かないでください。
- 質問に**直接答える**ことだけに集中してください。
- 複数のポイントがある場合は、箇点（・）で簡潔にまとめてください。
- コンテキストにない情報は絶対に使わないでください。
- 箇点は必ず1行に1つ。行の先頭は「・」。

### コンテキスト:
{context_text}

### 質問:
{query}

### 回答（簡潔に）:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", 
                     "content": "あなたは簡潔で正確な回答をするアシスタントです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,      # Lower temperature for more focused answers
                max_tokens=config.MAX_TOKENS,
                reasoning_effort="none",
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}") from e