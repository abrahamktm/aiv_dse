"""Tests for knowledge retriever (TF-IDF, chunking, caching)."""

import json
import os
import pytest

from aiv_dse.core.knowledge_retriever import (
    KnowledgeRetriever,
    _chunk_markdown,
    _build_tfidf_index,
    _query_tfidf,
)
from aiv_dse.llm.models import KnowledgeChunk


class TestChunking:
    def test_chunk_on_headers(self):
        text = "## Area\nReduce area tips.\n\n## Latency\nReduce latency tips."
        chunks = _chunk_markdown(text, "test.md", max_tokens=500)
        assert len(chunks) == 2
        assert "area" in chunks[0]["text"].lower()
        assert "latency" in chunks[1]["text"].lower()

    def test_chunk_large_section(self):
        """Large section is split on blank lines."""
        text = "## Big Section\n" + "\n\n".join(
            [f"Paragraph {i} with some content here." for i in range(20)]
        )
        chunks = _chunk_markdown(text, "test.md", max_tokens=100)
        assert len(chunks) > 1

    def test_chunk_source_metadata(self):
        text = "## DPO Modes\nDPO info here."
        chunks = _chunk_markdown(text, "hls_directives.md", max_tokens=500)
        assert "hls_directives.md" in chunks[0]["source"]
        assert "DPO Modes" in chunks[0]["source"]


class TestTFIDF:
    def test_retrieval_relevance(self):
        chunks = [
            {"text": "DPO_AUTO_ALL reduces area by 20 percent", "source": "a.md"},
            {"text": "Pipeline depth controls initiation interval", "source": "b.md"},
            {"text": "Resource sharing reduces area with power overhead", "source": "c.md"},
        ]
        index = _build_tfidf_index(chunks)
        results = _query_tfidf("reduce area", index, chunks, top_k=2)
        assert len(results) == 2
        # DPO and resource_sharing chunks should be more relevant
        texts = [r.text for r in results]
        assert any("area" in t.lower() for t in texts)

    def test_empty_query(self):
        chunks = [{"text": "Some content", "source": "a.md"}]
        index = _build_tfidf_index(chunks)
        results = _query_tfidf("", index, chunks, top_k=3)
        assert len(results) == 0

    def test_no_chunks(self):
        index = _build_tfidf_index([])
        results = _query_tfidf("anything", index, [], top_k=3)
        assert len(results) == 0


class TestKnowledgeRetriever:
    def test_build_from_directory(self):
        """Build index from the real knowledge/ directory."""
        kr = KnowledgeRetriever("knowledge")
        assert kr.chunk_count > 0

    def test_retrieve_area_query(self):
        kr = KnowledgeRetriever("knowledge")
        chunks = kr.retrieve("reduce area when unroll is high", top_k=3)
        assert len(chunks) >= 1
        assert all(isinstance(c, KnowledgeChunk) for c in chunks)
        assert all(c.score > 0 for c in chunks)

    def test_cache_created(self, tmp_path):
        """Cache file is created after build."""
        # Create a temp knowledge dir with a simple file
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / "test.md").write_text("## Test\nSome optimization tips.")
        (kdir / "sources.yaml").write_text(
            "confluence:\n  pages: []\nsettings:\n  chunk_max_tokens: 400\n"
            "  cache_ttl_days: 30\n  summarize_by_default: false\n"
        )

        kr = KnowledgeRetriever(str(kdir))
        assert kr.chunk_count >= 1
        assert os.path.exists(str(kdir / ".cache" / "index.json"))

    def test_query_builder(self):
        """build_query_from_violations produces meaningful query."""
        from aiv_dse.core.validator import ValidationResult
        from aiv_dse.llm.models import SynthesisParams

        result = ValidationResult(
            status="VETO",
            violations=[{
                "constraint_id": "area",
                "field": "area_units",
                "observed": 60000,
                "threshold": 50000,
                "severity": "WARNING",
                "action": "ESCALATE",
            }],
        )
        params = SynthesisParams(unroll_factor=8, dpo_mode="DPO_AUTO_ALL")
        query = KnowledgeRetriever.build_query_from_violations(result, params)
        assert "area" in query
        assert "unroll_factor=8" in query
        assert "DPO_AUTO_ALL" in query

    def test_sources_yaml_parsed(self):
        """Sources config is loaded correctly."""
        from aiv_dse.core.knowledge_retriever import _load_sources_config
        config = _load_sources_config("knowledge")
        assert "confluence" in config or "settings" in config
