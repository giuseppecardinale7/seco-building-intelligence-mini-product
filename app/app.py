"""
SECO Building Intelligence — Streamlit UI

Shows building footprints for Esch-sur-Alzette on an interactive map.
Click a building → see the structured fact card + AI pre-inspection risk brief.
On-demand brief generation for buildings not yet pre-processed.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import geopandas as gpd
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add pipeline to path so we can import config + generation helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import (
    GOLD, SILVER,
    EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION, TOP_K_CHUNKS,
)

GOLD_FILE   = GOLD / "buildings_esch_gold.geojson"
SILVER_FILE = SILVER / "buildings_esch.geojson"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SECO Building Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_buildings() -> gpd.GeoDataFrame | None:
    for path in (GOLD_FILE, SILVER_FILE):
        if path.exists():
            gdf = gpd.read_file(str(path))
            return gdf
    return None


@st.cache_resource
def load_retriever():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(CHROMA_COLLECTION)
        return model, collection
    except Exception:
        return None, None


# ── On-demand brief generation ────────────────────────────────────────────────

def generate_brief_on_demand(row: dict) -> tuple[str, list[dict]]:
    """Call the gold-generation helpers directly for one building."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
    from _gold_helpers import (
        generate_risk_brief,
        retrieve_regulation_snippets,
        _build_rag_query,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️  ANTHROPIC_API_KEY not set in .env", []

    model_name = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    embed_model, collection = load_retriever()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        snippets = []
        if embed_model and collection:
            query = _build_rag_query(row)
            snippets = retrieve_regulation_snippets(query, embed_model, collection)
        brief, citations = generate_risk_brief(row, snippets, client, model_name)
        return brief, citations
    except Exception as exc:
        return f"⚠️  Error generating brief: {exc}", []


# ── UI helpers ────────────────────────────────────────────────────────────────

def _era_badge(era: str | None) -> str:
    if not era or era == "unknown":
        return "⚪ Unknown era"
    if "pre-1945" in era:
        return "🔴 Pre-1945"
    if "1945" in era:
        return "🟠 1945–1969"
    if "1970" in era:
        return "🟡 1970–1989"
    if "1990" in era:
        return "🟢 1990–2009"
    return "🔵 2010+"


def _zone_badge(code: str | None) -> str:
    if not code:
        return "⚪ No PAG zone"
    if "HAB" in str(code):
        return f"🏠 {code}"
    if "MIX" in str(code) or "CEN" in str(code):
        return f"🏪 {code}"
    if "IND" in str(code) or "ECO" in str(code):
        return f"🏭 {code}"
    if "VER" in str(code) or "VERT" in str(code):
        return f"🌿 {code}"
    return f"📋 {code}"


def _format_risk_brief(text: str) -> None:
    """Render risk brief with colour-coded priority labels."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            st.write("")
            continue
        if "HIGH" in stripped.upper():
            st.markdown(f"🔴 {stripped}")
        elif "MEDIUM" in stripped.upper():
            st.markdown(f"🟡 {stripped}")
        elif "LOW" in stripped.upper():
            st.markdown(f"🟢 {stripped}")
        else:
            st.markdown(stripped)


# ── Map ───────────────────────────────────────────────────────────────────────

def build_map(gdf: gpd.GeoDataFrame, selected_id: str | None) -> "folium.Map":
    import folium
    from folium.plugins import MarkerCluster

    center = gdf.geometry.centroid.to_crs("EPSG:4326")
    lat_c = float(center.y.mean())
    lon_c = float(center.x.mean())

    m = folium.Map(location=[lat_c, lon_c], zoom_start=14,
                   tiles="CartoDB positron")

    # Colour by era
    era_colors = {
        "pre-1945": "#e74c3c",
        "1945–1969": "#e67e22",
        "1970–1989": "#f1c40f",
        "1990–2009": "#2ecc71",
        "2010+": "#3498db",
        "unknown": "#95a5a6",
    }

    def _style(feature):
        era = feature["properties"].get("osm_era", "unknown") or "unknown"
        color = era_colors.get(era, "#95a5a6")
        bid = feature["properties"].get("building_id", "")
        is_selected = bid == selected_id
        return {
            "fillColor": color,
            "color": "#2c3e50" if is_selected else "#7f8c8d",
            "weight": 3 if is_selected else 0.5,
            "fillOpacity": 0.85 if is_selected else 0.6,
        }

    def _popup(feature):
        p = feature["properties"]
        bid = p.get("building_id", "?")
        area = p.get("footprint_area_m2")
        area_str = f"{area:.0f} m²" if area else "—"
        zone = p.get("zone_code") or "—"
        era  = p.get("osm_era") or "unknown"
        has_brief = bool(p.get("risk_brief", ""))
        brief_icon = "✅" if has_brief else "⏳"
        return folium.Popup(
            f"<b>{bid}</b><br>"
            f"Area: {area_str}<br>"
            f"Zone: {zone}<br>"
            f"Era: {era}<br>"
            f"Brief: {brief_icon}",
            max_width=200,
        )

    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["building_id", "footprint_area_m2", "zone_code", "osm_era"],
            aliases=["ID", "Area (m²)", "PAG zone", "Era"],
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["building_id", "zone_code", "osm_era"],
            aliases=["ID", "Zone", "Era"],
        ),
    ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;">
    <b>Construction era</b><br>
    <span style="color:#e74c3c">■</span> Pre-1945<br>
    <span style="color:#e67e22">■</span> 1945–1969<br>
    <span style="color:#f1c40f">■</span> 1970–1989<br>
    <span style="color:#2ecc71">■</span> 1990–2009<br>
    <span style="color:#3498db">■</span> 2010+<br>
    <span style="color:#95a5a6">■</span> Unknown
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏗️ SECO Building Intelligence")
    st.markdown(
        "**Pre-inspection risk briefs for Esch-sur-Alzette, Luxembourg** "
        "| Data: BD-L-GeoBase (ACT 2026), national PAG, OpenStreetMap"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Building selector")
        gdf = load_buildings()

        if gdf is None:
            st.error(
                "No building data found.\n\n"
                "Run `make pipeline` first to download and process data."
            )
            st.stop()

        st.caption(f"{len(gdf)} buildings loaded · {GOLD_FILE.exists() and 'Gold' or 'Silver'} layer")

        # Search by building ID or zone
        search = st.text_input("Search by building ID or PAG zone code")
        if search:
            mask = (
                gdf["building_id"].astype(str).str.contains(search, case=False, na=False)
                | gdf.get("zone_code", pd.Series(dtype=str)).astype(str).str.contains(
                    search, case=False, na=False
                )
            )
            filtered = gdf[mask]
        else:
            filtered = gdf

        # Select a building
        bid_options = filtered["building_id"].astype(str).tolist()
        if not bid_options:
            st.warning("No buildings match the search.")
            selected_bid = None
        else:
            selected_bid = st.selectbox(
                f"Select building ({len(bid_options)} matched)",
                bid_options,
                format_func=lambda x: f"🏠 {x}",
            )

        st.divider()
        st.markdown("**Legend**")
        st.markdown("🔴 Pre-1945 · 🟠 1945–69 · 🟡 1970–89 · 🟢 1990–09 · 🔵 2010+")

        st.divider()
        st.markdown(
            "**About**\n\n"
            "This is a pre-inspection triage tool, not a substitute for a physical "
            "inspection. Energy certificates (CPE) and actual defect history are "
            "private data — this product uses construction era, roof geometry, and "
            "zoning as risk *proxies* only."
        )

    # ── Main panel ────────────────────────────────────────────────────────────
    col_map, col_card = st.columns([3, 2])

    with col_map:
        st.subheader("Building map")
        from streamlit_folium import st_folium
        m = build_map(gdf, selected_bid)
        map_data = st_folium(m, height=520, use_container_width=True)

        # Allow click-on-map to update selection
        clicked_id = None
        if map_data and map_data.get("last_object_clicked_tooltip"):
            tooltip_text = map_data["last_object_clicked_tooltip"]
            if "ID" in str(tooltip_text):
                # tooltip shows "ID\nBLD_000042\n..."
                lines = str(tooltip_text).split("\n")
                for j, ln in enumerate(lines):
                    if "ID" in ln and j + 1 < len(lines):
                        clicked_id = lines[j + 1].strip()
                        break

    with col_card:
        if selected_bid or clicked_id:
            bid = clicked_id or selected_bid
            row = gdf[gdf["building_id"].astype(str) == bid]
            if len(row) == 0:
                st.warning(f"Building {bid} not found.")
            else:
                row_dict = row.iloc[0].to_dict()
                _render_building_card(row_dict, gdf)
        else:
            st.info("Select a building in the sidebar or click one on the map.")

    # ── Stats footer ──────────────────────────────────────────────────────────
    with st.expander("Dataset statistics"):
        n_brief = gdf["risk_brief"].notna().sum() if "risk_brief" in gdf.columns else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Buildings", len(gdf))
        c2.metric("PAG zones matched",
                  int(gdf["zone_code"].notna().sum()) if "zone_code" in gdf.columns else "—")
        c3.metric("OSM era matched",
                  int((gdf.get("osm_era", pd.Series()) != "unknown").sum()))
        c4.metric("Risk briefs generated", n_brief)

        if "osm_era" in gdf.columns:
            era_counts = gdf["osm_era"].value_counts()
            st.bar_chart(era_counts)


def _render_building_card(row: dict, gdf: gpd.GeoDataFrame) -> None:
    """Render the building fact card + risk brief in the right column."""
    bid = row.get("building_id", "?")
    st.subheader(f"Building {bid}")

    # Fact card
    col1, col2 = st.columns(2)
    with col1:
        area = row.get("footprint_area_m2")
        st.metric("Footprint", f"{area:.0f} m²" if area else "—")
        st.metric("PAG zone", _zone_badge(row.get("zone_code")))
    with col2:
        h_lim = row.get("height_limit_m")
        st.metric("Height limit", f"{h_lim} m" if h_lim else "—")
        st.metric("Era", _era_badge(row.get("osm_era")))

    if row.get("osm_levels"):
        st.caption(f"Storeys (OSM): {row['osm_levels']}")
    if row.get("zone_label"):
        st.caption(f"Zone label: {row['zone_label']}")

    st.divider()

    # Risk brief
    st.subheader("Pre-inspection Risk Brief")
    brief = row.get("risk_brief")
    if brief and not str(brief).startswith("["):
        _format_risk_brief(brief)

        # Citations
        citations_raw = row.get("citations", "[]")
        try:
            citations = json.loads(citations_raw) if isinstance(citations_raw, str) else []
        except Exception:
            citations = []
        if citations:
            with st.expander("Sources used"):
                for c in citations:
                    st.markdown(f"- [{c.get('label','?')}]({c.get('url','#')})")
    else:
        st.info("No pre-generated brief. Generate one now:")
        if st.button("Generate risk brief", key=f"gen_{bid}"):
            with st.spinner("Calling Claude... (~5 sec)"):
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    st.error("ANTHROPIC_API_KEY not set in .env file.")
                else:
                    _generate_and_display(row, bid)


def _generate_and_display(row: dict, bid: str) -> None:
    """Generate a brief on-demand and display it."""
    try:
        # pipeline dir is already on sys.path (set at module import time)
        from _gold_helpers import (
            generate_risk_brief,
            retrieve_regulation_snippets,
            _build_rag_query,
        )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model_name = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        client = anthropic.Anthropic(api_key=api_key)

        embed_model, collection = load_retriever()
        snippets = []
        if embed_model and collection:
            query = _build_rag_query(row)
            snippets = retrieve_regulation_snippets(query, embed_model, collection)

        brief, citations = generate_risk_brief(row, snippets, client, model_name)
        _format_risk_brief(brief)

        if citations:
            with st.expander("Sources used"):
                for c in citations:
                    st.markdown(f"- [{c.get('label','?')}]({c.get('url','#')})")
    except Exception as exc:
        st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
