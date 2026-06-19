"""
Silver layer — clean, reproject, join, embed.

Steps:
  1. Unpack GeoBase zips → extract relevant GeoPackage layers
  2. Find Esch-sur-Alzette commune boundary (admin units)
  3. Filter buildings to Esch boundary, reproject EPSG:2169 → WGS84
  4. Filter PAG zones to Esch, reproject
  5. Spatial join: assign each building its PAG zone (zone code, label, height limit)
  6. Load OSM Overpass data → extract construction-era proxy per building footprint
  7. Merge all attributes into silver building GeoJSON
  8. Chunk + embed regulation HTML → persist in ChromaDB
"""

import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from config import (
    BRONZE, SILVER,
    COMMUNE_NAME,
    EPSG_SOURCE, EPSG_TARGET,
    PAG_ZONE_FIELD, PAG_HEIGHT_FIELD, PAG_CODE_FIELD,
    EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION,
    STATIC_REGULATION_CORPUS_FILE,
)

# ── Constants ────────────────────────────────────────────────────────────────
SILVER_BUILDINGS = SILVER / "buildings_esch.geojson"
SILVER_PAG       = SILVER / "pag_esch.geojson"
SILVER_METADATA  = SILVER / "pipeline_metadata.json"


# ── 1. Unpack GeoBase zips ───────────────────────────────────────────────────

def _unzip_to(zip_path: Path, dest_dir: Path) -> Path:
    """Unzip to dest_dir, return dest_dir. Skip if already done."""
    marker = dest_dir / ".unzipped"
    if marker.exists():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    marker.touch()
    return dest_dir


def _find_gpkg(directory: Path) -> Path:
    """Find the first .gpkg file in a directory tree."""
    matches = list(directory.rglob("*.gpkg"))
    if not matches:
        raise FileNotFoundError(f"No .gpkg found under {directory}")
    return matches[0]


# ── 2. Esch commune boundary ─────────────────────────────────────────────────

def load_commune_boundary() -> gpd.GeoDataFrame:
    print("[silver] Loading admin units → finding Esch-sur-Alzette boundary")
    admin_dir = SILVER / "_admin_units_raw"
    _unzip_to(BRONZE / "admin_units.zip", admin_dir)
    gpkg = _find_gpkg(admin_dir)

    # List available layers
    import fiona
    layers = fiona.listlayers(str(gpkg))
    print(f"         Layers in admin gpkg: {layers}")

    # Load communes layer (name varies; try common names)
    commune_layer = None
    for candidate in ("municipalities", "commune", "communes", "Municipalities",
                      "admunits", "AdminUnits"):
        if candidate in layers:
            commune_layer = candidate
            break
    if commune_layer is None:
        # Fall back to first layer
        commune_layer = layers[0]
        print(f"         [warn] guessing commune layer: {commune_layer}")

    gdf = gpd.read_file(str(gpkg), layer=commune_layer)
    print(f"         Loaded {len(gdf)} admin units from layer '{commune_layer}'")
    print(f"         Columns: {list(gdf.columns)}")
    print(f"         CRS: {gdf.crs}")

    # Find Esch row — try common name columns
    name_cols = [c for c in gdf.columns if "name" in c.lower() or "nom" in c.lower()
                 or "label" in c.lower() or "lib" in c.lower()]
    esch = None
    for col in name_cols:
        mask = gdf[col].astype(str).str.contains("Esch", case=False, na=False)
        candidates = gdf[mask]
        if len(candidates) > 0:
            esch = candidates[candidates[col].astype(str).str.contains("Alzette", case=False, na=False)]
            if len(esch) == 0:
                esch = candidates.iloc[[0]]
            esch = esch.iloc[[0]]
            print(f"         Found commune via column '{col}': {esch[col].values[0]}")
            break

    if esch is None or len(esch) == 0:
        raise ValueError(f"Could not locate Esch-sur-Alzette in admin units. "
                         f"Available name columns: {name_cols}")

    return esch


# ── 3. Buildings filtered + reprojected ──────────────────────────────────────

def load_buildings(esch_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("[silver] Loading buildings → filtering to Esch-sur-Alzette")
    bld_dir = SILVER / "_buildings_raw"
    _unzip_to(BRONZE / "buildings.zip", bld_dir)
    gpkg = _find_gpkg(bld_dir)

    import fiona
    layers = fiona.listlayers(str(gpkg))
    print(f"         Layers in buildings gpkg: {layers}")

    # Pick the footprint polygon layer
    bld_layer = None
    for candidate in ("buildings", "bati", "bati_2d", "Buildings",
                      "building", "BD_L_BATI"):
        if candidate in layers:
            bld_layer = candidate
            break
    if bld_layer is None:
        bld_layer = layers[0]
        print(f"         [warn] guessing building layer: {bld_layer}")

    # Read with bounding box filter (fast pre-filter in EPSG:2169)
    boundary_2169 = esch_boundary.to_crs(EPSG_SOURCE)
    bbox = tuple(boundary_2169.total_bounds)   # (minx, miny, maxx, maxy)
    print(f"         Bbox filter (EPSG:2169): {[round(x,0) for x in bbox]}")

    gdf = gpd.read_file(str(gpkg), layer=bld_layer, bbox=bbox)
    print(f"         Pre-filter: {len(gdf)} buildings in bbox")

    # Precise clip to commune boundary
    gdf = gdf.to_crs(EPSG_SOURCE)
    gdf = gpd.clip(gdf, boundary_2169)
    print(f"         After clip: {len(gdf)} buildings in Esch-sur-Alzette")

    # Reproject to WGS84
    gdf = gdf.to_crs(EPSG_TARGET)

    # Compute footprint area (must do in projected CRS)
    gdf_proj = gdf.to_crs(EPSG_SOURCE)
    gdf["footprint_area_m2"] = gdf_proj.geometry.area.round(1)

    # Keep useful columns only (drop redundant ACT internals)
    keep_cols = ["geometry", "footprint_area_m2"]
    for col in ("id", "ID", "objectid", "OBJECTID", "gid", "GID",
                "building_id", "BUILDING_ID", "fid", "FID"):
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "building_id"})
            keep_cols.append("building_id")
            break
    for col in ("year_built", "yearbuilt", "annee_constr", "construction_year"):
        if col in gdf.columns:
            keep_cols.append(col)

    gdf = gdf[[c for c in keep_cols if c in gdf.columns] +
               [c for c in gdf.columns if c not in keep_cols and c != "geometry"]]

    if "building_id" not in gdf.columns:
        gdf["building_id"] = [f"BLD_{i:06d}" for i in range(len(gdf))]

    gdf = gdf.reset_index(drop=True)
    return gdf


# ── 4. PAG zones filtered + reprojected ─────────────────────────────────────

PAG_ZONE_HEIGHT_MAP: dict[str, float] = {
    # Typical PAG height limits by zone category (metres, approximate)
    "HAB_1": 8.0,   "HAB_2": 12.0,  "HAB_3": 16.0,
    "MIX_u": 15.0,  "MIX_c": 10.0,
    "COM": 12.0,    "GARE": 20.0,
    "ECO_c1": 10.0, "ECO_c2": 10.0, "ECO_n": 10.0, "ECO_r": 12.0,
    "IND": 12.0,
    "SPEC": None,   "AGR": None,    "FOR": None,
    "JAR": 5.0,     "VERD": None,   "PARC": None,
    "REC": 8.0,     "BEP": None,
}

PAG_ZONE_LABEL_MAP: dict[str, str] = {
    "HAB_1": "Habitation (faible densité)",
    "HAB_2": "Habitation (densité moyenne)",
    "HAB_3": "Habitation (forte densité)",
    "MIX_u": "Zone mixte urbaine",
    "MIX_c": "Zone mixte commerciale",
    "COM": "Zone commerciale",
    "ECO_c1": "Zone d'activité économique (c1)",
    "ECO_c2": "Zone d'activité économique (c2)",
    "ECO_n": "Zone d'activité économique (nouvelles)",
    "ECO_r": "Zone d'activité économique (reconversion)",
    "IND": "Zone industrielle",
    "AGR": "Zone agricole",
    "FOR": "Zone forestière",
    "JAR": "Zone de jardins",
    "VERD": "Zone verte / espaces verts",
    "PARC": "Zone de parcs",
    "REC": "Zone de récréation",
    "SPEC": "Zone spéciale",
    "BEP": "Bande d'espaces protégés",
    "GARE": "Zone de gare / pôle d'échange",
}


def load_pag(esch_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("[silver] Loading national PAG → filtering to Esch-sur-Alzette")
    pag_dir = SILVER / "_pag_raw"
    _unzip_to(BRONZE / "pag_national.gpkg.zip", pag_dir)
    gpkg = _find_gpkg(pag_dir)

    # PAG_PAG_ZONAGE is the urbanistic zone polygons layer
    gdf_pag = gpd.read_file(str(gpkg), layer="PAG_PAG_ZONAGE")
    print(f"         PAG_PAG_ZONAGE: {len(gdf_pag)} features, CRS: {gdf_pag.crs}")

    # Filter to Esch commune code C059 (INS 059)
    esch_mask = gdf_pag["CODE_COM"] == "C059"
    gdf_esch = gdf_pag[esch_mask].copy()
    print(f"         Esch-sur-Alzette (C059) zones: {len(gdf_esch)}")
    print(f"         Zone categories: {sorted(gdf_esch['CATEGORIE'].unique())}")

    # Rename CATEGORIE → zone_code, derive label and height limit
    gdf_esch = gdf_esch.rename(columns={"CATEGORIE": "zone_code"})
    gdf_esch["zone_label"]    = gdf_esch["zone_code"].map(PAG_ZONE_LABEL_MAP)
    gdf_esch["height_limit_m"] = gdf_esch["zone_code"].map(PAG_ZONE_HEIGHT_MAP)

    # Reproject to WGS84
    gdf_esch = gdf_esch.to_crs(EPSG_TARGET)
    gdf_esch = gdf_esch.reset_index(drop=True)
    return gdf_esch


# ── 5. Spatial join buildings ← PAG zones ────────────────────────────────────

def join_pag(buildings: gpd.GeoDataFrame,
             pag: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("[silver] Spatial join: buildings ← PAG zones (centroid-in-polygon)")
    if len(pag) == 0:
        print("         [warn] PAG empty — skipping join, filling nulls")
        buildings["zone_code"]  = None
        buildings["zone_label"] = None
        buildings["height_limit_m"] = None
        return buildings

    pag_clean = pag[["geometry", "zone_code", "zone_label", "height_limit_m"]].copy()

    # Compute centroids in projected CRS for accuracy, then reproject for join
    centroids_proj = buildings.to_crs(EPSG_SOURCE).geometry.centroid
    bld_centroids = buildings.copy()
    bld_centroids.geometry = centroids_proj.to_crs(EPSG_TARGET)

    joined = gpd.sjoin(bld_centroids, pag_clean, how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")

    # Dedup: each building centroid should hit at most one zone; keep first match
    joined = joined[~joined.index.duplicated(keep="first")]

    # Restore original polygon geometries
    joined = joined.reindex(buildings.index)
    joined.geometry = buildings.geometry.values

    for col in ("zone_code", "zone_label", "height_limit_m"):
        buildings[col] = joined[col].values if col in joined.columns else None

    matched = buildings["zone_code"].notna().sum()
    print(f"         Matched {matched}/{len(buildings)} buildings to a PAG zone")
    return buildings


# ── 6. OSM construction-era proxy ────────────────────────────────────────────

def _osm_era_label(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year < 1945:
        return "pre-1945"
    if year < 1970:
        return "1945–1969"
    if year < 1990:
        return "1970–1989"
    if year < 2010:
        return "1990–2009"
    return "2010+"


def join_osm(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print("[silver] Loading OSM data → construction era proxy")
    osm_file = BRONZE / "osm_esch_buildings.json"
    if not osm_file.exists():
        print("         [warn] OSM file not found — skipping")
        buildings["osm_era"]    = None
        buildings["osm_levels"] = None
        return buildings

    raw = json.loads(osm_file.read_text())
    elements = raw.get("elements", [])

    # Build simple lookup: for each OSM way, get the tags
    node_coords: dict[int, tuple[float, float]] = {}
    way_records: list[dict] = []

    for el in elements:
        if el["type"] == "node":
            node_coords[el["id"]] = (el["lon"], el["lat"])
        elif el["type"] == "way" and "tags" in el:
            way_records.append(el)

    # Reconstruct way polygons and extract year / levels
    from shapely.geometry import Polygon

    osm_rows = []
    for way in way_records:
        node_ids = way.get("nodes", [])
        coords = [node_coords[n] for n in node_ids if n in node_coords]
        if len(coords) < 3:
            continue
        try:
            poly = Polygon(coords)
        except Exception:
            continue

        tags = way.get("tags", {})
        year = None
        for tag in ("start_date", "construction:date", "year_built"):
            val = tags.get(tag, "")
            m = re.search(r"\b(1[89]\d{2}|20[012]\d)\b", str(val))
            if m:
                year = int(m.group(1))
                break

        levels = None
        for tag in ("building:levels", "levels"):
            val = tags.get(tag)
            if val:
                try:
                    levels = int(float(str(val)))
                except ValueError:
                    pass
                break

        osm_rows.append({
            "geometry": poly,
            "osm_year": year,
            "osm_era": _osm_era_label(year),
            "osm_levels": levels,
            "osm_building_type": tags.get("building", "yes"),
        })

    if not osm_rows:
        print("         [warn] No valid OSM ways found")
        buildings["osm_era"]    = None
        buildings["osm_levels"] = None
        return buildings

    osm_gdf = gpd.GeoDataFrame(osm_rows, crs=EPSG_TARGET)
    print(f"         OSM buildings reconstructed: {len(osm_gdf)}")

    # Centroid join ACT buildings → OSM polygons
    centroids_wgs = buildings.geometry.to_crs(EPSG_TARGET) if buildings.crs.to_epsg() != 4326 \
        else buildings.geometry
    bld_centroids = buildings.copy()
    bld_centroids.geometry = buildings.geometry.centroid

    osm_cols = ["geometry", "osm_era", "osm_levels", "osm_building_type", "osm_year"]
    joined = gpd.sjoin(bld_centroids, osm_gdf[osm_cols],
                       how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")

    # Dedup: keep first OSM match per building
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.reindex(buildings.index)
    joined.geometry = buildings.geometry.values

    for col in ("osm_era", "osm_levels", "osm_building_type", "osm_year"):
        buildings[col] = joined[col].values if col in joined.columns else None

    matched = buildings["osm_era"].notna().sum()
    print(f"         OSM era matched: {matched}/{len(buildings)} buildings")
    return buildings


# ── 7. Regulation text → chunked → embedded → ChromaDB ──────────────────────

def _extract_text_from_html(html_bytes: bytes) -> str:
    """Extract visible text from HTML, strip boilerplate."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_bytes, "lxml")
    # Remove navigation, headers, footers, scripts, styles
    for tag in soup(["script", "style", "nav", "header", "footer",
                      "aside", "form", "button"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Normalise whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 20]
    return "\n".join(lines)


def _chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks of ~max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + "\n\n" + para).strip() if tail else para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 50]


def embed_regulations() -> int:
    """Embed regulations into ChromaDB.

    Strategy (in order):
    1. Static corpus (regulations_static.json) — always present, curated excerpts
       from public Luxembourg legislation. This is the primary RAG source.
    2. Scraped HTML pages from regulation websites — supplementary, if available.
    """
    print("[silver] Embedding regulation text → ChromaDB")

    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"         Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    # ── 1. Static corpus (primary) ────────────────────────────────────────────
    if STATIC_REGULATION_CORPUS_FILE.exists():
        static_docs = json.loads(STATIC_REGULATION_CORPUS_FILE.read_text())
        print(f"         Static corpus: {len(static_docs)} documents")
        for doc in static_docs:
            chunks = _chunk_text(doc["text"])
            if not chunks:
                chunks = [doc["text"]]
            embeddings = model.encode(chunks, show_progress_bar=False).tolist()
            ids = [f"{doc['id']}_s{i}" for i in range(len(chunks))]
            metadatas = [
                {"source_id": doc["id"], "label": doc["label"],
                 "url": doc.get("url", ""), "domain": doc.get("domain", "")}
                for _ in chunks
            ]
            collection.add(documents=chunks, embeddings=embeddings,
                           ids=ids, metadatas=metadatas)
            total_chunks += len(chunks)
        print(f"         Static corpus embedded: {total_chunks} chunks")
    else:
        print("         [warn] Static corpus not found")

    # ── 2. Scraped HTML (supplementary) ──────────────────────────────────────
    manifest_file = BRONZE / "regulations_manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        html_chunks = 0
        for src in manifest:
            html_path = Path(src.get("path", ""))
            if not html_path.exists():
                continue
            text = _extract_text_from_html(html_path.read_bytes())
            chunks = _chunk_text(text)
            if not chunks:
                continue
            embeddings = model.encode(chunks, show_progress_bar=False).tolist()
            ids = [f"{src['id']}_h{i}" for i in range(len(chunks))]
            metadatas = [
                {"source_id": src["id"], "label": src["label"],
                 "url": src["url"], "domain": src.get("domain", "")}
                for _ in chunks
            ]
            collection.add(documents=chunks, embeddings=embeddings,
                           ids=ids, metadatas=metadatas)
            total_chunks += len(chunks)
            html_chunks += len(chunks)
        if html_chunks > 0:
            print(f"         Scraped HTML embedded: {html_chunks} additional chunks")

    print(f"         Total chunks in ChromaDB: {total_chunks}")
    return total_chunks


# ── 8. Save silver outputs ───────────────────────────────────────────────────

def save_silver(buildings: gpd.GeoDataFrame, pag: gpd.GeoDataFrame,
                n_chunks: int) -> None:
    buildings.to_file(SILVER_BUILDINGS, driver="GeoJSON")
    print(f"[silver] Buildings saved → {SILVER_BUILDINGS.name}")
    print(f"         {len(buildings)} buildings, columns: {list(buildings.columns)}")

    if len(pag) > 0:
        pag.to_file(SILVER_PAG, driver="GeoJSON")
        print(f"[silver] PAG saved     → {SILVER_PAG.name}")

    meta = {
        "commune": "Esch-sur-Alzette",
        "n_buildings": len(buildings),
        "n_pag_zones": len(pag),
        "n_regulation_chunks": n_chunks,
        "source_crs": EPSG_SOURCE,
        "output_crs": EPSG_TARGET,
    }
    SILVER_METADATA.write_text(json.dumps(meta, indent=2))
    print(f"[silver] Metadata saved  → {SILVER_METADATA.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SILVER LAYER — transform, join, embed")
    print("=" * 60)

    esch_boundary = load_commune_boundary()
    buildings = load_buildings(esch_boundary)
    pag = load_pag(esch_boundary)
    buildings = join_pag(buildings, pag)
    buildings = join_osm(buildings)
    n_chunks = embed_regulations()
    save_silver(buildings, pag, n_chunks)

    print("\n[done] Silver transformation complete.")


if __name__ == "__main__":
    main()
