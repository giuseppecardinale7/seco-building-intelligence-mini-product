"""
Shared helpers for gold-layer brief generation — used by both
03_gold_generate.py (batch) and app/app.py (on-demand).

Supports two LLM providers:
  - Anthropic Claude (ANTHROPIC_API_KEY)
  - Google Gemini   (GOOGLE_API_KEY)  ← free tier available

The active provider is selected at call time by _get_client().
"""

import os
import re

from config import EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION, TOP_K_CHUNKS, RISK_BRIEF_SYSTEM


# ── Provider abstraction ──────────────────────────────────────────────────────

class _LLMClient:
    """Thin wrapper so gold code calls the same interface regardless of provider."""
    def generate(self, user_content: str) -> str:
        raise NotImplementedError


class _AnthropicClient(_LLMClient):
    def __init__(self, api_key: str, model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, user_content: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=RISK_BRIEF_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text.strip()


class _GeminiClient(_LLMClient):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=RISK_BRIEF_SYSTEM,
        )

    def generate(self, user_content: str) -> str:
        resp = self._model.generate_content(user_content)
        return resp.text.strip()


def get_llm_client() -> _LLMClient:
    """Return the appropriate LLM client based on available API keys."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key    = os.environ.get("GOOGLE_API_KEY", "")

    if anthropic_key and not anthropic_key.startswith("sk-ant-..."):
        model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        return _AnthropicClient(api_key=anthropic_key, model=model)

    if google_key and not google_key.startswith("AIza..."):
        model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        return _GeminiClient(api_key=google_key, model=model)

    raise EnvironmentError(
        "No LLM API key found. Set either:\n"
        "  ANTHROPIC_API_KEY  (get one at https://console.anthropic.com)\n"
        "  GOOGLE_API_KEY     (free tier at https://aistudio.google.com/apikey)\n"
        "in your .env file."
    )


# ── RAG retrieval ─────────────────────────────────────────────────────────────

def retrieve_regulation_snippets(query: str, model, collection,
                                  k: int = TOP_K_CHUNKS) -> list[dict]:
    embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    snippets = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        snippets.append({
            "text": doc,
            "label": meta.get("label", ""),
            "url": meta.get("url", ""),
            "domain": meta.get("domain", ""),
            "relevance": round(1 - dist, 3),
        })
    return snippets


def load_retriever():
    """Load SentenceTransformer + ChromaDB collection. Returns (model, collection)."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return None, None
    return model, collection


# ── Query + prompt building ───────────────────────────────────────────────────

def _build_rag_query(row: dict) -> str:
    parts = ["building inspection risk Luxembourg"]
    zone = row.get("zone_code", "")
    if zone:
        parts.append(f"zone {zone}")
    era = row.get("osm_era", "")
    if "pre-1945" in era or "1945" in era:
        parts.append("old building asbestos insulation heritage electrical")
    elif "1970" in era or "1989" in era:
        parts.append("energy insulation waterproofing renovation 1970s")
    elif "1990" in era or "2009" in era:
        parts.append("energy performance certificate renovation 1990s")
    btype = row.get("osm_building_type", "")
    if btype in ("residential", "apartments", "house"):
        parts.append("residential fire egress stairwell")
    elif btype in ("commercial", "retail", "office"):
        parts.append("commercial fire safety evacuation")
    if row.get("height_limit_m"):
        parts.append("height compliance urban planning PAG")
    parts.append("fire safety energy performance permit compliance amiante")
    return " ".join(parts)


def _build_facts_string(row: dict) -> str:
    lines = []
    if row.get("building_id"):
        lines.append(f"Building ID: {row['building_id']}")
    area = row.get("footprint_area_m2")
    if area:
        lines.append(f"Footprint area: {float(area):.0f} m²")
    zone = row.get("zone_code")
    if zone:
        label = row.get("zone_label") or ""
        lines.append(f"PAG zone: {zone} — {label}".strip(" —"))
    else:
        lines.append("PAG zone: not available")
    h_lim = row.get("height_limit_m")
    if h_lim is not None:
        lines.append(f"PAG height limit: {h_lim} m")
    era = row.get("osm_era")
    if era and era != "unknown":
        lines.append(f"Construction era (OSM proxy): {era}")
        if row.get("osm_year"):
            lines.append(f"  Approximate year: {row['osm_year']}")
    else:
        lines.append("Construction era: unknown (no OSM data)")
    levels = row.get("osm_levels")
    if levels:
        lines.append(f"Number of storeys (OSM): {levels}")
    btype = row.get("osm_building_type")
    if btype and str(btype) not in ("yes", "None", "nan"):
        lines.append(f"Building type (OSM): {btype}")
    return "\n".join(lines)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_risk_brief(row: dict, snippets: list[dict],
                        client: _LLMClient) -> tuple[str, list[dict]]:
    """Generate risk brief. Returns (brief_text, citations)."""
    facts = _build_facts_string(row)

    snippets_text = ""
    for i, s in enumerate(snippets, 1):
        snippets_text += (
            f"\n[SOURCE {i}] {s['label']}\n"
            f"URL: {s['url']}\n"
            f"{s['text'][:600]}\n"
        )

    user_content = (
        f"BUILDING FACTS:\n{facts}\n\n"
        f"RETRIEVED REGULATION SNIPPETS:{snippets_text}\n\n"
        "Generate the pre-inspection risk brief now."
    )

    brief_text = client.generate(user_content)
    citations = [
        {"label": s["label"], "url": s["url"], "domain": s["domain"]}
        for s in snippets
    ]
    return brief_text, citations
