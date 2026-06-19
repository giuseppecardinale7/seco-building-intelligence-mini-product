"""
SECO Building Intelligence — Streamlit UI

Shows building footprints for Esch-sur-Alzette on an interactive map.
Click a building → see the structured fact card + AI pre-inspection risk brief.
On-demand brief generation for buildings not yet pre-processed.
"""

import json
import os
import re
import sys
from pathlib import Path

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

def _zone_color(zone_code: str | None) -> str:
    """Map PAG zone code to a display colour."""
    if not zone_code or str(zone_code) in ("None", "nan", ""):
        return "#95a5a6"   # grey — no zone data
    z = str(zone_code).upper()
    if z.startswith("HAB"):
        return "#3498db"   # blue — residential
    if z.startswith("MIX") or z.startswith("CEN"):
        return "#9b59b6"   # purple — mixed / town centre
    if z.startswith("ECO") or z.startswith("IND"):
        return "#e67e22"   # orange — economic / industrial
    if z.startswith("AGR"):
        return "#27ae60"   # green — agricultural
    if z.startswith("VER") or z.startswith("VERT") or z.startswith("ESP"):
        return "#2ecc71"   # light green — green space
    return "#95a5a6"       # grey — other


LEGEND_HTML = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:10px;border-radius:8px;
            border:1px solid #ccc;font-size:12px;">
<b>PAG zone</b><br>
<span style="color:#3498db">■</span> Residential (HAB)<br>
<span style="color:#9b59b6">■</span> Mixed / Centre (MIX, CEN)<br>
<span style="color:#e67e22">■</span> Economic / Industrial (ECO, IND)<br>
<span style="color:#27ae60">■</span> Agricultural (AGR)<br>
<span style="color:#2ecc71">■</span> Green space (VER)<br>
<span style="color:#95a5a6">■</span> Other / unknown
</div>"""


def build_map(gdf: gpd.GeoDataFrame, selected_id: str | None) -> "folium.Map":
    """Render map with individual CircleMarkers for a 2000-building sample.

    The full 9 K polygon GeoJSON (~10 MB) crashes Leaflet.  GeoJson with
    marker= template also fails to render (fill=False default, style_function
    not overriding it reliably).  Individual folium.CircleMarker objects are
    verbose but guaranteed to render.  We cap at 2000 markers; the selected
    building is always included and drawn with its full polygon too.
    """
    import folium
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centroids = gdf.geometry.centroid

    lat_c = float(centroids.y.mean())
    lon_c = float(centroids.x.mean())

    m = folium.Map(location=[lat_c, lon_c], zoom_start=14,
                   tiles="CartoDB positron")

    # ── Sample buildings to display (cap at 2000 for browser perf) ───────────
    MAX_DOTS = 2000
    has_brief = gdf["risk_brief"].notna() if "risk_brief" in gdf.columns else pd.Series(False, index=gdf.index)
    priority = gdf[has_brief]
    remainder = gdf[~has_brief]
    n_rest = max(0, MAX_DOTS - len(priority))
    display = pd.concat([
        priority,
        remainder.sample(min(n_rest, len(remainder)), random_state=42),
    ])

    # ── Draw circle markers ───────────────────────────────────────────────────
    for idx, row in display.iterrows():
        bid  = str(row["building_id"])
        is_sel = bid == selected_id
        color = "#f39c12" if is_sel else _zone_color(row.get("zone_code"))
        zone  = row.get("zone_code") or "—"
        area  = row.get("footprint_area_m2")
        area_str = f"{area:.0f} m²" if pd.notna(area) else "—"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt = centroids[idx]

        folium.CircleMarker(
            location=[pt.y, pt.x],
            radius=6 if is_sel else 4,
            fill=True,
            fill_color=color,
            color="#2c3e50" if is_sel else "#ffffff",
            weight=2 if is_sel else 0.5,
            fill_opacity=1.0 if is_sel else 0.75,
            tooltip=f"ID: {bid}<br>Zone: {zone}<br>Area: {area_str}",
        ).add_to(m)

    # ── Selected building: full polygon ───────────────────────────────────────
    if selected_id:
        sel = gdf[gdf["building_id"].astype(str) == selected_id]
        if len(sel) > 0:
            folium.GeoJson(
                sel[["building_id", "geometry"]].__geo_interface__,
                style_function=lambda _: {
                    "fillColor": "#f39c12",
                    "color":     "#2c3e50",
                    "weight":    3,
                    "fillOpacity": 0.45,
                },
            ).add_to(m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pt = sel.geometry.centroid.iloc[0]
            m.location = [pt.y, pt.x]

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

        # Search — driven by map clicks via session state (key binds widget ↔ state)
        search = st.text_input(
            "Search by building ID or PAG zone code",
            key="search_input",
        )
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

        st.divider()
        st.markdown("**Map colour — PAG zone**")
        st.markdown("🔵 Residential (HAB)")
        st.markdown("🟣 Mixed / Town centre (MIX, CEN)")
        st.markdown("🟠 Economic / Industrial (ECO, IND)")
        st.markdown("🟢 Agricultural (AGR)")
        st.markdown("🟡 Green space (VER)")
        st.markdown("⚫ Other / no zone")

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


        # Map click → populate search bar via session state + rerun.
        # The tooltip from folium is HTML so we extract the BLD_ ID with regex.
        if map_data and map_data.get("last_object_clicked_tooltip"):
            match = re.search(r"BLD_\d+",
                              str(map_data["last_object_clicked_tooltip"]))
            if match:
                clicked_id = match.group(0)
                if clicked_id != st.session_state.get("search_input", ""):
                    st.session_state["search_input"] = clicked_id
                    st.rerun()

    with col_card:
        if selected_bid:
            row = gdf[gdf["building_id"].astype(str) == selected_bid]
            if len(row) == 0:
                st.warning(f"Building {selected_bid} not found.")
            else:
                _render_building_card(row.iloc[0].to_dict(), gdf)
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
            with st.spinner("Generating risk brief... (~5 sec)"):
                _generate_and_display(row, bid)


def _generate_and_display(row: dict, bid: str) -> None:
    """Generate a brief on-demand and display it."""
    try:
        # pipeline dir is already on sys.path (set at module import time)
        from _gold_helpers import (
            generate_risk_brief,
            retrieve_regulation_snippets,
            get_llm_client,
            _build_rag_query,
        )
        llm_client = get_llm_client()

        embed_model, collection = load_retriever()
        snippets = []
        if embed_model and collection:
            query = _build_rag_query(row)
            snippets = retrieve_regulation_snippets(query, embed_model, collection)

        brief, citations = generate_risk_brief(row, snippets, llm_client)
        _format_risk_brief(brief)

        if citations:
            with st.expander("Sources used"):
                for c in citations:
                    st.markdown(f"- [{c.get('label','?')}]({c.get('url','#')})")
    except EnvironmentError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
