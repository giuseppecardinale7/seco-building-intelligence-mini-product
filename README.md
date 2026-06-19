# SECO Building Intelligence

A pre-inspection triage tool for building inspectors and asset managers.
Type an address or click a building → get a structured fact sheet assembled from
Luxembourg public data, plus an AI-generated risk brief with regulation citations.

Pilot area: **Esch-sur-Alzette**, Luxembourg's second city.

---

## What problem does this solve, and for whom?

**User**: A SECO building inspector or asset manager preparing for a site visit or
due-diligence review.

**Problem**: Before inspecting a building, the inspector manually pieces together
scattered facts — cadastral footprint, zoning constraints, applicable regulations —
from disconnected public sources. This wastes time and risks missing red flags before
ever stepping on site.

**Solution**: Type or pick a building → the pipeline assembles a per-building fact
sheet from heterogeneous Luxembourg public sources → an LLM synthesises those
structured facts with retrieved regulation snippets to produce a plain-language
*pre-inspection risk brief* with per-flag citations.

Example output:

```
1. [HIGH] Building era 1970–1989 — Energy insulation likely non-compliant with
   current RGD 23 nov. 2016 requirements. Check wall U-values and roof insulation
   on site. [Legilux — RGD 23 nov. 2016 performance énergétique]

2. [MEDIUM] PAG zone HAB_1 — verify building height against 12 m limit recorded in
   PAG. No construction year on record; measure from ground floor sill.
   [Guichet.lu — Conditions d'autorisation de construire]

3. [LOW] Footprint 320 m² with unknown roof type — flat roof is statistically
   likely for this era; check waterproofing and drainage.
   [ITM — Prévention incendie dans les bâtiments]

Overall: Medium-risk profile; priority is energy compliance and height verification.
```

---

## Why is this relevant to SECO?

SECO's stated challenge is turning "huge volumes of technical data that remain largely
underexploited" into actionable intelligence. This product attacks the pre-inspection
phase — exactly where SECO inspectors invest time before any value-generating work
happens on site. The AI component does genuine synthesis (structured facts + regulation
retrieval → judgement-flavoured brief), not a bolted-on chatbot. The data engineering
demonstrates the cross-source joins SECO would need to scale this to Belgium and
Luxembourg's full building stock.

---

## Data sources

| # | Source | Format | What it provides | Why used |
|---|--------|--------|-----------------|----------|
| 1 | **BD-L-GeoBase buildings** (ACT, Apr 2026) | GeoPackage, EPSG:2169 | Building footprint polygons, national coverage | Official Luxembourg building layer — authoritative geometry |
| 2 | **BD-L-GeoBase addresses** (ACT, Apr 2026) | GeoPackage, EPSG:2169 | Address points linked to buildings | Address → building lookup in UI |
| 3 | **BD-L-GeoBase administrative units** (ACT, Apr 2026) | GeoPackage, EPSG:2169 | Commune boundaries | Filter all datasets to Esch pilot area |
| 4 | **National PAG GeoPackage** (ACT, Jun 2026) | GeoPackage, national | Zoning rules for all communes: zone code, zone label, height limits | Zoning constraints are a core pre-inspection risk signal |
| 5 | **OpenStreetMap / Overpass API** | JSON, WGS84 | Building tags: `start_date`, `building:levels`, `building` type | Construction-era proxy — the one attribute missing from ACT data |
| 6 | **Regulation HTML** (guichet.lu, itm.public.lu, legilux.public.lu) | HTML → chunked text | Building permit rules, fire safety, energy performance standards | RAG corpus — gives the AI real regulation text to cite |

All sources are public, no authentication required, and reproducible by running
`make pipeline`.

---

## Technical decisions and trade-offs

### Pipeline architecture — Medallion

```
Bronze  →  Silver  →  Gold
  ↓           ↓          ↓
Raw files   Cleaned,   Per-building
(as-is)     joined,    brief + citations
            embedded   (GeoJSON)
```

- **Bronze**: files stored exactly as downloaded, with a provenance JSON log
  (`data/bronze/provenance.json`) recording URL, size, and fetch timestamp.
  Skips already-downloaded files for safe re-runs.
- **Silver**: unzips and reads GeoPackage layers; filters to Esch commune boundary
  using a bbox pre-filter then a polygon clip (fast even on national files);
  reprojects from EPSG:2169 (LUREF) to WGS84; spatial-joins buildings ← PAG zones
  (centroid-in-polygon); OSM-joins for construction era; chunks and embeds regulation
  text into ChromaDB.
- **Gold**: for each building, runs a RAG query → retrieves top-4 regulation
  snippets → calls Claude to generate the risk brief. Checkpoints every 50 buildings
  so a failed run can resume.

### Why Streamlit, not React

React is SECO's preferred stack, but for a 5-day solo build, Streamlit eliminates
build tooling, bundling, and state management overhead. The architecture is cleanly
separable — the gold GeoJSON file is the API contract, so the UI layer can be
swapped for a React/Next.js frontend without touching the pipeline.

### Why sentence-transformers (local embeddings)

Using a local model (`paraphrase-multilingual-MiniLM-L12-v2`) means:
- No embedding API key required for reviewers
- No per-query cost for the RAG retrieval step
- Handles French regulation text well
Claude is used only for generation (the expensive, high-quality step).

### Why ChromaDB

Lightweight, persistent, zero-infrastructure vector store — correct for a single-node
MVP. In production this would be replaced by pgvector (on top of the existing PostGIS
stack for the geospatial data) or a managed service.

### Key trade-off accepted: pre-generation vs. on-demand

Gold briefs are pre-generated for a sample of 200 buildings (configurable via
`--sample`). This makes the UI snappy. On-demand generation is available for
non-pre-generated buildings but adds ~3 s latency per building. In production, briefs
would be regenerated nightly via a pipeline schedule.

---

## Known limitations — stated explicitly

### What the product CANNOT tell you

- **Energy Performance Certificates (CPE / Energiepass)**: Individual EPCs are private
  data. The product uses *construction era* as an energy risk proxy, not actual
  measured energy class.
- **Actual inspection history**: Prior defect findings, remediation records, and
  maintenance logs are not open data. The risk brief is triage intelligence, not a
  replacement for an inspection.
- **Roof type**: BD-L-GeoBase 2D footprints do not include roof geometry. We use
  statistical inference (era + building type → likely flat/pitched) as a proxy. The
  3D BATI3D dataset (CityGML, LOD 2.2) has roof geometry but is structured per-commune
  and would add significant pipeline complexity for the MVP.

### What the AI brief IS and IS NOT

- IS: a structured reasoning step that combines public facts with applicable
  regulation text to surface plausible risk flags for an expert to verify on site.
- IS NOT: a professional inspection report, legal advice, or compliance certification.
  The inspector's on-site judgement remains the authoritative source.

These limitations are documented in the UI sidebar and in every generated brief.

---

## What would go to production tomorrow vs. what gets thrown away

### Ship tomorrow
- The Medallion pipeline structure (Bronze → Silver → Gold) — correct abstraction
  for a data product that refreshes as source data updates
- The RAG-over-regulations pattern — regulation text changes slowly; re-embedding
  monthly is sufficient
- The ChromaDB collection + sentence-transformers embeddings — works well for the
  French regulation corpus, zero ops overhead

### Throw away
- Streamlit UI → replace with Next.js + MapLibre GL for a proper GIS experience
- Local ChromaDB → replace with pgvector (co-located with the PostGIS building table)
- The OSM construction-era proxy → replace with actual cadastral year-of-construction
  once SECO/ACT data sharing agreement is in place
- Pre-generated gold GeoJSON file → replace with a PostGIS + FastAPI backend that
  serves building facts on-demand and generates briefs server-side

---

## If I had 3 more months

1. **Full Luxembourg coverage** — run the pipeline for all 100 communes, not just
   Esch. The architecture already handles national data; it's a parameter change.
2. **Roof type from BATI3D** — parse the CityGML per-commune, extract roof surface
   normal vectors → classify flat / pitched / mansard. This turns the energy-risk
   proxy from "era guess" into "geometry fact."
3. **Historical orthophoto change detection** — diff ACT orthophotos (2001–2023)
   using a lightweight CNN to detect facade changes, extensions, or new rooftop
   structures not reflected in current cadastral data. This is the "computer vision"
   component that the brief mentions but the MVP leaves as a stretch goal.
4. **Regulation corpus expansion** — add Belgian NBN standards (SECO is Belgian),
   Eurocode structural loads, and the full ITM fire-safety circular corpus. More
   regulation depth = better risk flags.
5. **Feedback loop** — inspectors mark which AI risk flags proved correct. Fine-tune
   the retrieval query and prompt using those signals. This is how the product gets
   better over time without manual curation.
6. **Private data integration** — negotiate access to CPE data (Energiepass) via the
   MEV portal. With actual energy class per building, the energy risk flag moves from
   proxy to fact.

---

## Running the project

### Requirements
- Python 3.12
- `make` (macOS: comes with Xcode CLT)
- An Anthropic API key (for gold layer brief generation)

### Setup

```bash
git clone <repo-url>
cd seco-project

# Create venv + install dependencies
make setup

# Configure API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### Run the pipeline

```bash
# Download raw data (~330 MB total, takes ~5–10 min)
make bronze

# Clean, reproject, join, embed regulations
make silver

# Generate AI risk briefs (200 buildings by default, ~2 min)
make gold

# Launch the app
make app
```

Or run all pipeline steps in sequence:

```bash
make pipeline && make app
```

### Running individual pipeline steps

```bash
cd pipeline

# Bronze: re-run is safe — skips already-downloaded files
../.venv/bin/python 01_bronze_ingest.py

# Silver: idempotent — re-running rebuilds silver from bronze
../.venv/bin/python 02_silver_transform.py

# Gold: resumes from last checkpoint, generates for new buildings only
../.venv/bin/python 03_gold_generate.py --sample 50 --delay 0.5
```

---

## Repository structure

```
seco-project/
├── pipeline/
│   ├── config.py               # Source URLs, paths, AI settings
│   ├── 01_bronze_ingest.py     # Download raw data with provenance logging
│   ├── 02_silver_transform.py  # Reproject, join, embed regulations
│   ├── 03_gold_generate.py     # Per-building RAG + Claude risk brief
│   └── _gold_helpers.py        # Shared RAG/Claude helpers (used by app too)
├── app/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── bronze/                 # Raw downloads (gitignored, ~330 MB)
│   ├── silver/                 # Cleaned GeoJSON + ChromaDB (gitignored)
│   └── gold/                   # Gold GeoJSON with risk briefs (gitignored)
├── requirements.txt
├── .env.example
├── Makefile
└── README.md
```

---

*SECO Take-Home Challenge — Giuseppe Cardinale — June 2026*
