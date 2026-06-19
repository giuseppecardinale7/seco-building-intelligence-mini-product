"""
Pipeline configuration — source URLs, paths, pilot commune settings.
All sources are public Luxembourg open data (data.public.lu / overpass-api.de).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ────────────────────────────────────────────────────────
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
GOLD   = ROOT / "data" / "gold"

for d in (BRONZE, SILVER, GOLD):
    d.mkdir(parents=True, exist_ok=True)

# ── Pilot commune ────────────────────────────────────────────────────────────
# Esch-sur-Alzette: Luxembourg's 2nd city, fully covered by ACT/PAG datasets.
# Deliberately NOT the City of Luxembourg (VDL handles its data separately).
COMMUNE_NAME = "Esch-sur-Alzette"
COMMUNE_CODE = "059"   # INS code used in PAG file naming convention

# WGS84 bounding box for Esch-sur-Alzette (used for OSM / fallback queries)
# [south, west, north, east]
ESCH_BBOX_WGS84 = (49.47, 5.96, 49.52, 6.01)

# ── BD-L-GeoBase (ACT, April 2026) — data.public.lu dataset 66d82e8cd69ea79173e85f42 ──
GEOBASE_BASE = "https://download.data.public.lu/resources/bd-l-geobase"
BUILDINGS_URL  = f"{GEOBASE_BASE}/20260401-011245/buildings-20260401.zip"
ADDRESSES_URL  = f"{GEOBASE_BASE}/20260401-011010/addresses-20260401.zip"
ADMIN_URL      = f"{GEOBASE_BASE}/20260401-011013/administrativeunits-20260401.zip"

# ── National PAG GeoPackage — data.public.lu dataset 5e318f0af176a17e68ca547a ──
# All "version 2011" PAGs in force, updated 2026-06-15. 266 MB compressed.
PAG_URL = "https://download.data.public.lu/resources/pag-geometries-de-tous-les-pag-version-2011-en-vigueur/20260615-023539/pag.gpkg.zip"

# ── OSM Overpass API — construction-era proxy ────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass QL: buildings in Esch bbox, requesting all tags for era/height hints
OVERPASS_QUERY = """
[out:json][timeout:90];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out body;
>;
out skel qt;
""".format(s=ESCH_BBOX_WGS84[0], w=ESCH_BBOX_WGS84[1],
           n=ESCH_BBOX_WGS84[2], e=ESCH_BBOX_WGS84[3])

# ── Regulation sources (for RAG corpus) ──────────────────────────────────────
REGULATION_SOURCES = [
    {
        "id": "itm_construction_sante",
        "url": "https://itm.public.lu/fr/securite-sante-travail/construction.html",
        "label": "ITM — Construction: sécurité et santé au travail",
        "domain": "structural_safety",
    },
    {
        "id": "itm_chantiers",
        "url": "https://itm.public.lu/fr/securite-sante-travail/chantiers-temporaires-mobiles.html",
        "label": "ITM — Chantiers temporaires et mobiles",
        "domain": "structural_safety",
    },
]

# Static regulation corpus (curated excerpts from public LU legislation).
# Used as primary RAG content — always available, no scraping needed.
# Sources: RGD 2016/11/23, RGD 2011/08/10, PAG general provisions, ITM circulars.
STATIC_REGULATION_CORPUS_FILE = ROOT / "pipeline" / "regulations_static.json"

# ── Silver / Gold field names ─────────────────────────────────────────────────
EPSG_SOURCE = "EPSG:2169"   # LUREF / Luxembourg Transverse Mercator
EPSG_TARGET = "EPSG:4326"   # WGS84 for Streamlit/Folium

# PAG zone fields expected in the national GeoPackage
PAG_ZONE_FIELD   = "lib_zone"   # human-readable zone label
PAG_HEIGHT_FIELD = "hmax"       # maximum height in metres (may be null)
PAG_CODE_FIELD   = "cod_zone"   # short zone code e.g. HAB_1, MIX_U

# ── AI settings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
CHROMA_PATH     = str(SILVER / "chroma_db")
CHROMA_COLLECTION = "lu_regulations"
TOP_K_CHUNKS    = 4   # regulation snippets retrieved per building

RISK_BRIEF_SYSTEM = """You are a pre-inspection assistant for SECO Group, an independent technical control and engineering group in Luxembourg. Your job is to produce a concise PRE-INSPECTION RISK BRIEF for a building inspector or asset manager preparing for a site visit.

Rules:
- Write in English.
- Base every risk flag ONLY on the building facts provided and the retrieved regulation snippets.
- For each flag cite the source in [brackets].
- Use priority labels HIGH / MEDIUM / LOW.
- Be specific: mention the building characteristic that triggers the flag.
- Do NOT invent facts not present in the input.
- End with a one-sentence Overall Assessment.
- Format: plain text with numbered flags, no markdown headers."""
