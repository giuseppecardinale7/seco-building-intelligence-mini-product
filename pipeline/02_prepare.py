"""
Step 2 — build the buildings dataset for Esch-sur-Alzette.

What this does:
  1. Loads building footprints from BD-L-GeoBase (official cadastral data).
  2. Filters to Esch-sur-Alzette using the municipality boundary.
  3. Loads PAG zone polygons (the urban zoning plan).
  4. Spatial-joins each building to its PAG zone.
  5. Saves a clean GeoJSON that the Streamlit app reads.

Run time: ~2 minutes (the PAG file is 266 MB).
"""

import warnings
import zipfile
from pathlib import Path

import geopandas as gpd

warnings.filterwarnings("ignore", message=".*CRS.*")

DATA    = Path(__file__).resolve().parent.parent / "data"
OUT     = DATA / "buildings_esch.geojson"

ESCH_MUNICIPALITY = "Esch-sur-Alzette"
ESCH_PAG_CODE     = "C059"          # commune code used inside the PAG file


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_gpkg(zip_path: Path) -> Path:
    """Unzip a GeoPackage and return the path to the .gpkg file."""
    with zipfile.ZipFile(zip_path) as z:
        gpkg_name = next(n for n in z.namelist() if n.endswith(".gpkg"))
        out_path  = DATA / Path(gpkg_name).name
        if not out_path.exists():
            print(f"  extracting {zip_path.name} ...")
            z.extract(gpkg_name, DATA)
            # zipfile may put it in a subdirectory — move it to DATA root
            extracted = DATA / gpkg_name
            if extracted != out_path:
                extracted.rename(out_path)
        return out_path


# ── load buildings ────────────────────────────────────────────────────────────

buildings_gpkg = extract_gpkg(DATA / "buildings.zip")

print("Loading buildings from BD-L-GeoBase ...")
buildings = gpd.read_file(buildings_gpkg, layer="BU_Building")
buildings = buildings.to_crs("EPSG:4326")           # → standard lat/lon

# Find Esch boundary and keep only buildings inside it
admin = gpd.read_file(buildings_gpkg, layer="AU_AdministrativeMunicipality")
admin = admin.to_crs("EPSG:4326")
esch_boundary = admin.loc[admin["Name"] == ESCH_MUNICIPALITY, "geometry"].union_all()

# A building is "in Esch" if its centroid is inside the boundary
esch = buildings[buildings.geometry.centroid.within(esch_boundary)].copy()
print(f"  {len(esch)} buildings in {ESCH_MUNICIPALITY}")


# ── load PAG zones ────────────────────────────────────────────────────────────

pag_gpkg = extract_gpkg(DATA / "pag.zip")

print("Loading PAG zoning plan ...")
pag = gpd.read_file(pag_gpkg, layer="PAG_PAG_ZONAGE")
pag = pag[pag["CODE_COM"] == ESCH_PAG_CODE].to_crs("EPSG:4326")
print(f"  {len(pag)} zone polygons for Esch")


# ── spatial join ──────────────────────────────────────────────────────────────
# For each building, find which PAG zone polygon contains its centroid.

print("Joining buildings with PAG zones ...")
centroids = gpd.GeoDataFrame(esch, geometry=esch.geometry.centroid, crs="EPSG:4326")

joined = gpd.sjoin(
    centroids,
    pag[["CATEGORIE", "geometry"]],
    how="left",
    predicate="within",
)
# Drop duplicates that arise when a centroid falls on a zone boundary
joined = joined[~joined.index.duplicated(keep="first")]

esch["zone_code"]  = joined["CATEGORIE"]
esch["zone_label"] = joined.get("LIB_ZONE", joined.get("LIBELLE", ""))   # label column differs by version


# ── clean up and save ─────────────────────────────────────────────────────────

esch = esch.reset_index(drop=True)
esch["building_id"]       = [f"BLD_{i:06d}" for i in range(len(esch))]
esch["footprint_area_m2"] = (esch.geometry.area * (111_320 ** 2)).round(1)

# Only keep columns the app needs
esch = esch[["building_id", "zone_code", "zone_label", "footprint_area_m2", "geometry"]]

esch.to_file(OUT, driver="GeoJSON")

n_zoned = int(esch["zone_code"].notna().sum())
print(f"\n  saved {len(esch)} buildings  →  {OUT}")
print(f"  PAG zone matched: {n_zoned}/{len(esch)} ({100*n_zoned/len(esch):.1f}%)")
print("\nDone. Open the app with:  streamlit run app/app.py")
