"""
PHANTOM PLATFORM — RAG Connector Plugin
========================================
Retrieves top-k relevant document passages from local vector index
and injects retrieved context into the prompt prior to generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from phantom.plugins.base import BasePlugin, GenerationContext


class RAGPlugin(BasePlugin):
    name = "rag-connector"
    version = "1.0.0"

    def __init__(self, db_path: Optional[str] = None, top_k: int = 3):
        self.db_path = Path(db_path or (Path.home() / ".phantom" / "rag" / "default"))
        self.top_k = top_k
        self.documents: List[Dict[str, str]] = []
        self._load_local_docs()

    def _load_local_docs(self):
        self.db_path.mkdir(parents=True, exist_ok=True)
        # Load any text files in the RAG store
        for p in self.db_path.glob("*.txt"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self.documents.append({"title": p.stem, "content": f.read()})
            except Exception:
                pass

    def add_document(self, title: str, content: str):
        self.documents.append({"title": title, "content": content})

    async def pre_generate(self, prompt: str, context: GenerationContext) -> str:
        """Search documents for terms in prompt and prepend relevant passages."""
        if not self.documents:
            return prompt

        # Lightweight TF-IDF / keyword similarity for local RAG
        keywords = set(prompt.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc["content"].lower().split())
            overlap = len(keywords.intersection(doc_words))
            if overlap > 0:
                scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_passages = [d[1] for d in scored_docs[: self.top_k]]

        if not top_passages:
            return prompt

        context_str = "\n\n".join(
            [f"[Document: {p['title']}]\n{p['content']}" for p in top_passages]
        )
        augmented_prompt = f"""Use the following retrieved context to answer the prompt.

--- RETRIEVED CONTEXT ---
{context_str}
-------------------------

Prompt: {prompt}"""
        return augmented_prompt
