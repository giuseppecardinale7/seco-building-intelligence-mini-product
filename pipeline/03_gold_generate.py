"""
Gold layer — per-building risk briefs via RAG + Claude.

For each building in the silver dataset:
  1. Retrieve top-K relevant regulation chunks from ChromaDB
  2. Call Claude (claude-haiku-4-5 by default) to generate a risk brief
  3. Attach the brief + citation list to the building record

The gold GeoJSON is what the Streamlit app reads.
Pre-generates briefs for --sample buildings (default 200); on-demand
generation is also available from the app.
"""

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

from config import SILVER, GOLD
from _gold_helpers import (
    load_retriever,
    retrieve_regulation_snippets,
    generate_risk_brief,
    get_llm_client,
    _build_rag_query,
)

load_dotenv()

GOLD_BUILDINGS   = GOLD / "buildings_esch_gold.geojson"
SILVER_BUILDINGS = SILVER / "buildings_esch.geojson"


def main(sample: int | None = 200, delay: float = 0.3) -> None:
    print("=" * 60)
    print("GOLD LAYER — RAG + Claude risk brief generation")
    print("=" * 60)

    if not SILVER_BUILDINGS.exists():
        raise FileNotFoundError(
            f"{SILVER_BUILDINGS} not found. Run 02_silver_transform.py first."
        )
    gdf = gpd.read_file(str(SILVER_BUILDINGS))
    print(f"[gold] Loaded {len(gdf)} silver buildings")

    # Resume support — only skip buildings that already have a real brief
    if GOLD_BUILDINGS.exists():
        gold_gdf = gpd.read_file(str(GOLD_BUILDINGS))
        has_brief = (
            gold_gdf["risk_brief"].notna() & (gold_gdf["risk_brief"] != "")
            if "risk_brief" in gold_gdf.columns
            else pd.Series(False, index=gold_gdf.index)
        )
        already_done = set(gold_gdf.loc[has_brief, "building_id"].astype(str))
        print(f"[gold] Resuming — {len(already_done)} briefs already generated")
    else:
        gold_gdf = None
        already_done: set[str] = set()

    todo = gdf[~gdf["building_id"].astype(str).isin(already_done)]
    if sample and len(todo) > sample:
        todo = todo.sample(sample, random_state=42)
    print(f"[gold] Will generate briefs for {len(todo)} buildings")

    if len(todo) == 0:
        print("[gold] Nothing to do — all buildings already have briefs.")
        return

    llm_client = get_llm_client()
    print(f"[gold] LLM provider: {type(llm_client).__name__}")

    embed_model, collection = load_retriever()
    n_docs = collection.count() if collection else 0
    if n_docs == 0:
        print("[gold] [warn] ChromaDB empty — briefs will have no regulation context")

    new_rows = []

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        row_dict = row.to_dict()
        bid = row_dict.get("building_id", f"idx_{i}")
        print(f"  [{i}/{len(todo)}] {bid}", end="  ", flush=True)

        try:
            snippets = (
                retrieve_regulation_snippets(_build_rag_query(row_dict),
                                            embed_model, collection)
                if n_docs > 0 else []
            )
            brief, citations = generate_risk_brief(row_dict, snippets, llm_client)
            row_dict["risk_brief"] = brief
            row_dict["citations"]  = json.dumps(citations, ensure_ascii=False)
            print("✓")
        except Exception as exc:
            print(f"Error: {exc}")
            row_dict["risk_brief"] = f"[Generation failed: {exc}]"
            row_dict["citations"]  = "[]"

        new_rows.append(row_dict)

        if delay > 0:
            time.sleep(delay)

        if i % 50 == 0 or i == len(todo):
            _checkpoint(new_rows, gold_gdf, gdf.crs)
            print(f"  [checkpoint] saved {len(new_rows)} new briefs")

    _checkpoint(new_rows, gold_gdf, gdf.crs)
    print(f"\n[done] Gold layer complete → {GOLD_BUILDINGS}")


def _checkpoint(new_rows: list[dict], gold_gdf, crs) -> None:
    if not new_rows:
        return
    new_gdf = gpd.GeoDataFrame(new_rows, crs=crs)
    if gold_gdf is not None and len(gold_gdf) > 0:
        combined = gpd.GeoDataFrame(
            pd.concat([gold_gdf, new_gdf], ignore_index=True), crs=crs
        )
    else:
        combined = new_gdf
    combined.to_file(str(GOLD_BUILDINGS), driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200,
                        help="Max buildings to process (default 200; 0=all)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (default 0.3)")
    args = parser.parse_args()
    main(sample=args.sample or None, delay=args.delay)
