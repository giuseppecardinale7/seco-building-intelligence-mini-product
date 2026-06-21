# SECO Building Intelligence

A pre-inspection triage tool for Esch-sur-Alzette, Luxembourg. Select a building on the map and receive a structured risk brief generated from public cadastral and zoning data, cross-referenced against a curated set of Luxembourg regulation excerpts.

Built as a take-home challenge for SECO Group — June 2026.

---

## Overview

Before a site visit, inspectors typically piece together relevant facts from multiple disconnected public sources. This tool consolidates that process: building footprint, PAG zoning classification, and applicable regulations are assembled automatically, and an LLM produces a prioritised risk brief with per-flag regulation citations.

The output is a triage aid — it surfaces what to verify on site, not a substitute for the inspection itself.

---

## How it works

1. ~9,000 building footprints for Esch-sur-Alzette are displayed on an interactive map, colour-coded by PAG zone (residential, mixed-use, industrial, etc.)
2. Selecting a building — by clicking a dot or searching by ID / zone code — displays its footprint area and zoning classification
3. Clicking "Generate brief" sends the building facts alongside 10 regulation excerpts to Llama 3.1 (via Groq), which returns a structured brief with HIGH / MEDIUM / LOW risk flags and specific regulation citations

---

## Data sources

- **Building footprints**: BD-L-GeoBase (Administration du Cadastre et de la Topographie, April 2026) — national GeoPackage in EPSG:2169 (LUREF), reprojected to WGS84
- **Zoning**: National PAG GeoPackage (data.public.lu, June 2026) — each building centroid is spatially joined to its PAG zone polygon to retrieve zone code and label
- **Regulations**: 10 curated excerpts in `regulations.json` covering energy performance (RGD 2016), fire safety, asbestos obligations (pre-1998 buildings), electrical standards (RGIE / IEC 60364), building permits (Loi 2004), and flat roof waterproofing (DTU 43)

All sources are publicly available with no authentication required.

---

## Setup

```bash
git clone <repo>
cd seco-project

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# Set GROQ_API_KEY — free tier at console.groq.com
```

---

## Running

### Rebuilding the data pipeline

```bash
# Download raw data (~330 MB) from data.public.lu
.venv/bin/python pipeline/01_download.py

# Filter to Esch-sur-Alzette, spatial-join PAG zones, output GeoJSON
.venv/bin/python pipeline/02_prepare.py
```

`02_prepare.py` takes approximately 2 minutes, most of which is reading the 266 MB PAG file.

### Launching the app

```bash
.venv/bin/streamlit run app/app.py
```

The app also resolves `data/silver/buildings_esch.geojson` as a fallback, so the pipeline does not need to be re-run if the prepared data already exists.

---

## Repository structure

```
seco-project/
├── pipeline/
│   ├── 01_download.py      # fetch raw data from data.public.lu
│   └── 02_prepare.py       # filter to Esch, spatial-join PAG zones, save GeoJSON
├── app/
│   └── app.py              # Streamlit UI and Groq LLM integration
├── regulations.json         # regulation excerpts included in every LLM prompt
├── .env.example
└── requirements.txt
```

---

## Design decisions

**Regulations in the prompt, not a vector database.** With only 10 short excerpts, including them all in every prompt is simpler and more reliable than a retrieval step. A production version with a larger regulation corpus would use semantic search (e.g. ChromaDB + multilingual embeddings) to retrieve only the most relevant excerpts per building.

**CircleMarkers on a 2,000-building sample.** Rendering all 9,000 building polygons as GeoJSON in a Leaflet iframe exceeds browser memory. A sample of 2,000 individual CircleMarkers is a practical workaround for the prototype; a production frontend (React + MapLibre GL + vector tiles) would handle the full dataset without this constraint.

**Groq / Llama 3.1 for generation.** The free tier is sufficient for a demo with on-demand generation. The model, temperature, and max tokens are configurable via `.env`.

---

## Known limitations

- **No construction year**: OSM `start_date` tags are largely absent for Esch, so era-specific risk flags (pre-1970 wiring, pre-1998 asbestos risk) cannot be inferred from current data. Actual year-of-construction would require a data-sharing agreement with ACT.
- **No energy performance data**: Individual EPCs (Energiepass / APE) are private records. The brief cannot reference actual energy class, only zoning-based proxies.
- **No roof geometry**: BD-L-GeoBase provides 2D footprints only. Roof type (flat vs. pitched) would require the BATI3D CityGML dataset (LOD 2.2), which adds significant pipeline complexity.

## What a production version would change

- Replace Streamlit with a React / MapLibre GL frontend serving a PostGIS + FastAPI backend
- Move ChromaDB vector store to pgvector, co-located with the PostGIS building table
- Add a nightly pipeline run to refresh briefs as source data updates
- Expand regulation coverage to Belgian NBN standards and the full ITM corpus
- Integrate a feedback loop so inspectors can flag which risk flags proved accurate on site
- Containerise the pipeline and app with Docker; orchestrate with Docker Compose for local development and Kubernetes for production
- CI/CD pipeline (GitHub Actions or equivalent) covering linting, data validation tests, and automated deployment on merge to main
- Integrate SECO's internal data — prior inspection reports, defect history, client records — to enrich the brief with building-specific context that public data cannot provide

---

*Giuseppe Cardinale — June 2026*
