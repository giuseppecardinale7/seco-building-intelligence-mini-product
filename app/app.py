import json
import os
import re
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_folium import st_folium

warnings.filterwarnings("ignore", message=".*CRS.*")
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
REGULATIONS_FILE = ROOT / "regulations.json"

_candidates = [
    ROOT / "data" / "buildings_esch.geojson",
    ROOT / "data" / "silver" / "buildings_esch.geojson",
]
BUILDINGS_FILE = next(p for p in _candidates if p.exists())

ZONE_COLORS = {
    "HAB": "#3498db",
    "MIX": "#9b59b6",
    "CEN": "#9b59b6",
    "ECO": "#e67e22",
    "IND": "#e67e22",
    "AGR": "#27ae60",
    "VER": "#2ecc71",
}

SYSTEM_PROMPT = """You are a pre-inspection assistant for SECO Group, an independent
building inspection company in Luxembourg.

Given the known facts about a building and a set of regulation excerpts, write a
concise PRE-INSPECTION RISK BRIEF with numbered risk flags (HIGH / MEDIUM / LOW).
For each flag, cite the regulation in [brackets] and explain why that specific
building characteristic triggers it.
End with one sentence: Overall Assessment.
Write in English. Be specific — don't flag risks that aren't supported by the facts."""


@st.cache_data
def load_buildings():
    return gpd.read_file(str(BUILDINGS_FILE))

@st.cache_data
def load_regulations():
    with open(REGULATIONS_FILE) as f:
        return json.load(f)


def zone_color(zone_code):
    if not zone_code or str(zone_code) in ("None", "nan"):
        return "#95a5a6"
    return ZONE_COLORS.get(str(zone_code)[:3].upper(), "#95a5a6")


def build_map(buildings, selected_id):
    m = folium.Map(location=[49.495, 5.985], zoom_start=14, tiles="CartoDB positron")

    sample = buildings.sample(min(2000, len(buildings)), random_state=42)

    for _, row in sample.iterrows():
        bid = str(row["building_id"])
        is_sel = bid == selected_id
        pt = row.geometry.centroid
        zone = row.get("zone_code") or "—"
        area = row.get("footprint_area_m2")
        area_str = f"{area:.0f} m²" if pd.notna(area) else "—"

        folium.CircleMarker(
            location=[pt.y, pt.x],
            radius=6 if is_sel else 4,
            fill=True,
            fill_color="#f39c12" if is_sel else zone_color(zone),
            color="#2c3e50" if is_sel else "#ffffff",
            weight=2 if is_sel else 0.5,
            fill_opacity=1.0 if is_sel else 0.75,
            tooltip=f"ID: {bid}<br>Zone: {zone}<br>Area: {area_str}",
        ).add_to(m)

    if selected_id is not None:
        sel = buildings[buildings["building_id"] == selected_id]
        if len(sel) > 0:
            folium.GeoJson(
                sel[["geometry"]].__geo_interface__,
                style_function=lambda _: {
                    "fillColor": "#f39c12",
                    "color": "#2c3e50",
                    "weight": 2,
                    "fillOpacity": 0.3,
                },
            ).add_to(m)

    return m


def generate_brief(row, regulations):
    facts = (
        f"Building ID : {row['building_id']}\n"
        f"PAG zone    : {row.get('zone_code') or 'unknown'}\n"
        f"Zone label  : {row.get('zone_label') or '—'}\n"
        f"Floor area  : {row.get('footprint_area_m2') or '?'} m²\n"
    )

    reg_text = "\n\n".join(f"[{r['label']}]\n{r['text']}" for r in regulations)

    user_message = (
        f"BUILDING FACTS:\n{facts}\n\n"
        f"REGULATION EXCERPTS:\n{reg_text}\n\n"
        "Write the pre-inspection risk brief now."
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content


st.set_page_config(
    page_title="SECO Building Intelligence",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ SECO Building Intelligence")
st.caption(
    "Pre-inspection risk briefs · Esch-sur-Alzette, Luxembourg · "
    "Data: BD-L-GeoBase (ACT 2026) + national PAG"
)

buildings = load_buildings()
regulations = load_regulations()

with st.sidebar:
    st.header("Select a building")

    search = st.text_input("Search by ID or zone code", key="search_input")

    if search:
        mask = (
            buildings["building_id"].str.contains(search, case=False, na=False)
            | buildings["zone_code"].astype(str).str.contains(search, case=False, na=False)
        )
        options = buildings.loc[mask, "building_id"].tolist()
    else:
        options = buildings["building_id"].tolist()

    selected = (
        st.selectbox(f"{len(options)} buildings", options, format_func=lambda x: f"🏠 {x}")
        if options else None
    )

    st.divider()
    st.markdown("**Map colours — PAG zone**")
    st.markdown("🔵 Residential (HAB)")
    st.markdown("🟣 Mixed / Town centre (MIX, CEN)")
    st.markdown("🟠 Economic / Industrial (ECO, IND)")
    st.markdown("🟢 Agricultural (AGR)")
    st.markdown("🟡 Green space (VER)")
    st.markdown("⚫ Other / unknown")

    st.divider()
    st.caption(
        "Prototype — uses publicly available data only. "
        "Private data (energy certificates, inspection history) not included."
    )

col_map, col_card = st.columns([3, 2])

with col_map:
    st.subheader("Map")
    m = build_map(buildings, selected)
    map_data = st_folium(m, height=500, use_container_width=True)

    tooltip = str(map_data.get("last_object_clicked_tooltip") or "")
    match = re.search(r"BLD_\d+", tooltip)
    if match and match.group(0) != st.session_state.get("search_input", ""):
        st.session_state["search_input"] = match.group(0)
        st.rerun()

with col_card:
    if selected is None:
        st.info("Click a dot on the map or search in the sidebar.")
    else:
        row = buildings[buildings["building_id"] == selected].iloc[0].to_dict()

        st.subheader(f"Building {selected}")

        col1, col2 = st.columns(2)
        col1.metric("Floor area", f"{row.get('footprint_area_m2') or 0:.0f} m²")
        col2.metric("PAG zone", row.get("zone_code") or "—")
        if row.get("zone_label"):
            st.caption(f"Zone: {row['zone_label']}")

        st.divider()
        st.subheader("Pre-inspection Risk Brief")
        st.caption("Generated by Llama 3.1 (Groq) from building facts + regulation excerpts.")

        if st.button("⚡ Generate brief", type="primary"):
            with st.spinner("Generating…"):
                brief = generate_brief(row, regulations)
            st.write(brief)

with st.expander("Dataset overview"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Buildings (Esch-sur-Alzette)", len(buildings))
    c2.metric("With PAG zone", int(buildings["zone_code"].notna().sum()))
    c3.metric("Data sources", 2)
    st.caption("Pipeline: BD-L-GeoBase buildings → filter to Esch → spatial join with PAG zones → GeoJSON")
