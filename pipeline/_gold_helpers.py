"""
Shared helpers for gold-layer brief generation — used by both
03_gold_generate.py (batch) and app/app.py (on-demand).
"""

import re

import anthropic

from config import EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION, TOP_K_CHUNKS, RISK_BRIEF_SYSTEM


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
    if row.get("zone_code"):
        parts.append(f"zone {row['zone_code']}")
    era = row.get("osm_era", "")
    if "pre-1945" in era or "1945" in era:
        parts.append("old building asbestos insulation heritage")
    elif "1970" in era or "1989" in era:
        parts.append("energy insulation waterproofing renovation")
    elif "1990" in era or "2009" in era:
        parts.append("energy performance certificate renovation")
    btype = row.get("osm_building_type", "")
    if btype in ("residential", "apartments", "house"):
        parts.append("residential fire egress stairwell")
    elif btype in ("commercial", "retail", "office"):
        parts.append("commercial fire safety evacuation")
    if row.get("height_limit_m"):
        parts.append("height compliance urban planning")
    parts.append("fire safety energy performance permit compliance")
    return " ".join(parts)


def _build_facts_string(row: dict) -> str:
    lines = []
    if row.get("building_id"):
        lines.append(f"Building ID: {row['building_id']}")
    if row.get("footprint_area_m2"):
        lines.append(f"Footprint area: {row['footprint_area_m2']:.0f} m²")
    if row.get("zone_code"):
        label = row.get("zone_label") or ""
        lines.append(f"PAG zone: {row['zone_code']} — {label}".strip(" —"))
    else:
        lines.append("PAG zone: not available")
    if row.get("height_limit_m") is not None:
        lines.append(f"PAG height limit: {row['height_limit_m']} m")
    osm_era = row.get("osm_era")
    if osm_era and osm_era != "unknown":
        lines.append(f"Construction era (OSM proxy): {osm_era}")
        if row.get("osm_year"):
            lines.append(f"  Approximate year: {row['osm_year']}")
    else:
        lines.append("Construction era: unknown (no OSM data)")
    if row.get("osm_levels"):
        lines.append(f"Number of storeys (OSM): {row['osm_levels']}")
    btype = row.get("osm_building_type")
    if btype and btype not in ("yes", None):
        lines.append(f"Building type (OSM): {btype}")
    return "\n".join(lines)


# ── Claude call ───────────────────────────────────────────────────────────────

def generate_risk_brief(row: dict, snippets: list[dict],
                        client: anthropic.Anthropic,
                        model_name: str) -> tuple[str, list[dict]]:
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

    response = client.messages.create(
        model=model_name,
        max_tokens=600,
        system=RISK_BRIEF_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    brief_text = response.content[0].text.strip()
    citations = [
        {"label": s["label"], "url": s["url"], "domain": s["domain"]}
        for s in snippets
    ]
    return brief_text, citations
