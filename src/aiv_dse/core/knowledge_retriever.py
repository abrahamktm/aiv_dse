"""RAG knowledge retriever with TF-IDF, multi-source ingestion, and caching.

Supports:
  - Local markdown (.md), text (.txt), and PDF (.pdf) files
  - Confluence pages (URLs in sources.yaml, fetched at build time)
  - Persistent caching to avoid re-ingestion on every run
  - Optional LLM-powered summarization for token efficiency

Chunking strategy:
  - Split on ## headers (markdown sections)
  - If a section > chunk_max_tokens, split on blank lines
  - Each chunk gets source file + section title as metadata

TF-IDF retrieval uses Python stdlib only (no sklearn).
"""

import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from aiv_dse.llm.models import KnowledgeChunk


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _chunk_markdown(text: str, source: str, max_tokens: int = 400) -> List[Dict[str, str]]:
    """Split markdown text into chunks on ## headers, then on blank lines."""
    sections = re.split(r'\n(?=## )', text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Extract title from first line if it's a header
        lines = section.split('\n')
        title = lines[0].lstrip('#').strip() if lines[0].startswith('#') else ""
        chunk_source = f"{source} > {title}" if title else source

        if _estimate_tokens(section) <= max_tokens:
            chunks.append({"text": section, "source": chunk_source})
        else:
            # Split on blank lines
            paragraphs = re.split(r'\n\s*\n', section)
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if _estimate_tokens(current + "\n\n" + para) > max_tokens and current:
                    chunks.append({"text": current.strip(), "source": chunk_source})
                    current = para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append({"text": current.strip(), "source": chunk_source})
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    return re.findall(r'[a-z0-9_]+', text.lower())


def _build_tfidf_index(chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build a TF-IDF index over chunks. Returns serializable index dict."""
    # Document frequency
    n_docs = len(chunks)
    df: Counter = Counter()
    doc_tfs = []

    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        tf = Counter(tokens)
        doc_tfs.append(tf)
        for token in set(tokens):
            df[token] += 1

    # IDF
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((n_docs + 1) / (freq + 1)) + 1.0

    # TF-IDF vectors (sparse, stored as dicts)
    tfidf_vectors = []
    for tf in doc_tfs:
        vec = {}
        for term, count in tf.items():
            vec[term] = count * idf.get(term, 0.0)
        tfidf_vectors.append(vec)

    return {
        "idf": idf,
        "vectors": tfidf_vectors,
    }


def _query_tfidf(
    query: str,
    index: Dict[str, Any],
    chunks: List[Dict[str, str]],
    top_k: int = 3,
) -> List[KnowledgeChunk]:
    """Retrieve top-K chunks by TF-IDF cosine similarity."""
    tokens = _tokenize(query)
    query_tf = Counter(tokens)
    idf = index["idf"]

    # Query vector
    query_vec = {}
    for term, count in query_tf.items():
        query_vec[term] = count * idf.get(term, 0.0)

    # Cosine similarity with each doc
    scores = []
    for i, doc_vec in enumerate(index["vectors"]):
        dot = sum(query_vec.get(t, 0.0) * doc_vec.get(t, 0.0) for t in query_vec)
        mag_q = math.sqrt(sum(v ** 2 for v in query_vec.values())) or 1.0
        mag_d = math.sqrt(sum(v ** 2 for v in doc_vec.values())) or 1.0
        scores.append((i, dot / (mag_q * mag_d)))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores[:top_k]:
        if score > 0:
            results.append(KnowledgeChunk(
                text=chunks[idx]["text"],
                source=chunks[idx]["source"],
                score=round(score, 4),
            ))
    return results


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf_file(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF ingestion. pip install pdfplumber")
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _load_sources_config(knowledge_dir: str) -> Dict[str, Any]:
    """Load sources.yaml if it exists."""
    config_path = os.path.join(knowledge_dir, "sources.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _fetch_confluence_page(url: str) -> str:
    """Fetch a Confluence page via curl. Returns extracted text or empty string."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            # Strip HTML tags (basic)
            text = re.sub(r'<[^>]+>', ' ', result.stdout)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


# ---------------------------------------------------------------------------
# KnowledgeRetriever
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """TF-IDF keyword retriever with multi-source ingestion and persistent caching."""

    def __init__(
        self,
        knowledge_dir: str,
        cache_dir: Optional[str] = None,
    ):
        self._knowledge_dir = knowledge_dir
        self._cache_dir = cache_dir or os.path.join(knowledge_dir, ".cache")
        self._chunks: List[Dict[str, str]] = []
        self._index: Dict[str, Any] = {}
        self._loaded = False

        # Try loading from cache
        if self._cache_exists():
            self._load_cache()
        else:
            self.build_index()

    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, "index.json")

    def _cache_exists(self) -> bool:
        cache_path = self._cache_path()
        if not os.path.exists(cache_path):
            return False
        # Check TTL from sources.yaml
        config = _load_sources_config(self._knowledge_dir)
        ttl_days = config.get("settings", {}).get("cache_ttl_days", 30)
        mtime = os.path.getmtime(cache_path)
        age_days = (time.time() - mtime) / 86400
        return age_days < ttl_days

    def _load_cache(self) -> None:
        """Load chunks and index from cache."""
        with open(self._cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = data["chunks"]
        self._index = data["index"]
        self._loaded = True

    def _save_cache(self) -> None:
        """Save chunks and index to cache."""
        os.makedirs(self._cache_dir, exist_ok=True)
        data = {"chunks": self._chunks, "index": self._index}
        with open(self._cache_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def build_index(
        self,
        summarize: bool = False,
        settings: Any = None,
    ) -> None:
        """Ingest all sources, chunk, and build TF-IDF index.

        Args:
            summarize: If True and settings provided, compress chunks via LLM.
            settings: LLMSettings for summarization (optional).
        """
        config = _load_sources_config(self._knowledge_dir)
        max_tokens = config.get("settings", {}).get("chunk_max_tokens", 400)

        all_chunks: List[Dict[str, str]] = []

        # 1. Local files
        if os.path.isdir(self._knowledge_dir):
            for fname in sorted(os.listdir(self._knowledge_dir)):
                fpath = os.path.join(self._knowledge_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                if fname.endswith((".md", ".txt")):
                    text = _load_text_file(fpath)
                    all_chunks.extend(_chunk_markdown(text, fname, max_tokens))
                elif fname.endswith(".pdf"):
                    text = _load_pdf_file(fpath)
                    all_chunks.extend(_chunk_markdown(text, fname, max_tokens))

        # 2. Confluence pages
        confluence_pages = config.get("confluence", {}).get("pages", [])
        for page in confluence_pages:
            url = page.get("url", "") if isinstance(page, dict) else str(page)
            label = page.get("label", url) if isinstance(page, dict) else url
            if url:
                text = _fetch_confluence_page(url)
                if text:
                    all_chunks.extend(
                        _chunk_markdown(text, f"confluence:{label}", max_tokens)
                    )

        # 3. Optional LLM summarization
        if summarize and settings and all_chunks:
            all_chunks = self._summarize_chunks(all_chunks, settings)

        # 4. Build TF-IDF index
        self._chunks = all_chunks
        if all_chunks:
            self._index = _build_tfidf_index(all_chunks)
        else:
            self._index = {"idf": {}, "vectors": []}
        self._loaded = True

        # 5. Save cache
        self._save_cache()

    def _summarize_chunks(
        self,
        chunks: List[Dict[str, str]],
        settings: Any,
    ) -> List[Dict[str, str]]:
        """Compress verbose chunks using LLM. One-time cost at build time."""
        try:
            from aiv_dse.llm.config import get_anthropic_client
            client = get_anthropic_client(settings)
        except Exception:
            return chunks  # Can't summarize, return originals

        summarized = []
        for chunk in chunks:
            if _estimate_tokens(chunk["text"]) < 200:
                summarized.append(chunk)
                continue
            try:
                response = client.messages.create(
                    model=settings.model_name,
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Compress this HLS optimization knowledge into concise "
                            "bullet points. Preserve all parameter names, values, "
                            "and relationships. Remove filler words.\n\n"
                            f"{chunk['text']}"
                        ),
                    }],
                )
                compressed = response.content[0].text
                summarized.append({
                    "text": compressed,
                    "source": chunk["source"],
                })
            except Exception:
                summarized.append(chunk)  # Keep original on failure
        return summarized

    def retrieve(self, query: str, top_k: int = 3) -> List[KnowledgeChunk]:
        """Retrieve top-K chunks most relevant to the query."""
        if not self._loaded or not self._chunks:
            return []
        return _query_tfidf(query, self._index, self._chunks, top_k)

    @staticmethod
    def build_query_from_violations(
        result: Any,
        params: Any,
    ) -> str:
        """Build a retrieval query from current violations and params.

        Args:
            result: ValidationResult with violations list.
            params: Current SynthesisParams.
        """
        parts = []

        # Add violation info
        violations = getattr(result, "violations", []) or []
        for v in violations:
            cid = v.get("constraint_id", "")
            field = v.get("field", "")
            observed = v.get("observed", "")
            threshold = v.get("threshold", "")
            if cid:
                parts.append(f"{cid} violated")
            if observed and threshold:
                pct = round((observed - threshold) / threshold * 100)
                parts.append(f"{field} {pct}% over")

        # Add key param values
        if hasattr(params, "unroll_factor"):
            parts.append(f"unroll_factor={params.unroll_factor}")
        if hasattr(params, "dpo_mode") and params.dpo_mode != "none":
            parts.append(f"dpo_mode={params.dpo_mode}")
        if hasattr(params, "flatten") and params.flatten:
            parts.append("flatten=True")

        return " ".join(parts) if parts else "HLS optimization"

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
