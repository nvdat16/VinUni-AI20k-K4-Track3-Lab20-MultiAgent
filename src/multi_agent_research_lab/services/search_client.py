"""Search client abstraction for ResearcherAgent."""

import json
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client backed by the offline lab corpus."""

    def __init__(self, corpus_dir: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.corpus_dir = corpus_dir or repo_root / "ai_agent_offline_research_corpus_v2" / "topics"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        The starter lab ships a self-contained offline corpus, so this implementation avoids
        internet search and retrieves relevant embedded articles/source summaries by keyword.
        """

        query_terms = self._tokenize(query)
        scored: list[tuple[int, SourceDocument]] = []
        for topic_path in sorted(self.corpus_dir.glob("*.json")):
            topic = json.loads(topic_path.read_text(encoding="utf-8"))
            topic_name = str(topic.get("topic", {}).get("name", topic_path.stem))
            scored.extend(self._score_topic(topic, topic_name, topic_path.name, query_terms))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for score, document in scored[:max_results] if score > 0]

    def _score_topic(
        self,
        topic: dict[str, Any],
        topic_name: str,
        filename: str,
        query_terms: set[str],
    ) -> list[tuple[int, SourceDocument]]:
        knowledge_base = topic.get("knowledge_base", {})
        scored: list[tuple[int, SourceDocument]] = []

        for article in knowledge_base.get("knowledge_articles", []):
            article_id = str(article.get("article_id", "article"))
            title = str(article.get("title", article_id))
            content = str(article.get("content", ""))
            citation_id = article_id
            scored.append(
                (
                    self._score(query_terms, title, content),
                    SourceDocument(
                        title=f"{topic_name}: {title}",
                        snippet=self._snippet(content),
                        metadata={
                            "topic": topic_name,
                            "filename": filename,
                            "citation_id": citation_id,
                            "source_type": "knowledge_article",
                        },
                    ),
                )
            )

        for document in knowledge_base.get("source_documents", []):
            document_id = str(document.get("document_id", document.get("source_id", "source")))
            title = str(document.get("title", document_id))
            content = self._document_text(document)
            citation_id = str(document.get("citation_label", document_id))
            scored.append(
                (
                    self._score(query_terms, title, content),
                    SourceDocument(
                        title=f"{topic_name}: {title}",
                        url=document.get("url") or document.get("provenance_url"),
                        snippet=self._snippet(content),
                        metadata={
                            "topic": topic_name,
                            "filename": filename,
                            "citation_id": citation_id,
                            "source_type": "source_document",
                            "is_synthetic": bool(document.get("is_synthetic", False)),
                        },
                    ),
                )
            )

        return scored

    def _document_text(self, document: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("summary", "content", "abstract", "description", "full_text"):
            value = document.get(key)
            if isinstance(value, str):
                parts.append(value)
        for key in ("key_claims", "limitations", "relevant_findings", "key_takeaways"):
            value = document.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
        return "\n".join(parts) or json.dumps(document, ensure_ascii=False)

    def _score(self, query_terms: set[str], title: str, content: str) -> int:
        title_terms = self._tokenize(title)
        content_terms = self._tokenize(content)
        return 3 * len(query_terms & title_terms) + len(query_terms & content_terms)

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3}

    def _snippet(self, text: str, max_chars: int = 500) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."
