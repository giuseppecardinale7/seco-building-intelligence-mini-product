"""
Bronze layer — raw data ingestion.

Downloads and saves source data exactly as received, with provenance metadata.
No transformation; transformation happens in 02_silver_transform.py.

Sources:
  1. BD-L-GeoBase buildings  (ACT, April 2026) — GeoPackage zip, ~39 MB
  2. BD-L-GeoBase addresses  (ACT, April 2026) — GeoPackage zip, ~13 MB
  3. BD-L-GeoBase admin units(ACT, April 2026) — GeoPackage zip, ~9 MB
  4. National PAG GeoPackage  (ACT, June 2026) — all communes, ~266 MB
  5. OpenStreetMap via Overpass API — building tags (construction era proxy)
  6. Regulation HTML pages    (guichet.lu, itm.public.lu, legilux.public.lu)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    BRONZE,
    BUILDINGS_URL,
    ADDRESSES_URL,
    ADMIN_URL,
    PAG_URL,
    OVERPASS_URL,
    OVERPASS_QUERY,
    REGULATION_SOURCES,
)

PROVENANCE_FILE = BRONZE / "provenance.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_provenance() -> dict:
    if PROVENANCE_FILE.exists():
        return json.loads(PROVENANCE_FILE.read_text())
    return {}


def _save_provenance(prov: dict) -> None:
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=2, ensure_ascii=False))


def _already_fetched(prov: dict, key: str) -> bool:
    return prov.get(key, {}).get("status") == "ok"


def _record(prov: dict, key: str, path: Path, url: str, size_bytes: int) -> None:
    prov[key] = {
        "status": "ok",
        "url": url,
        "local_path": str(path),
        "size_bytes": size_bytes,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_provenance(prov)


def _download(url: str, dest: Path, label: str, prov: dict, key: str,
              chunk_size: int = 1 << 20) -> Path:
    """Stream-download url → dest with a progress bar. Skip if already done."""
    if _already_fetched(prov, key) and dest.exists():
        print(f"  [skip] {label} — already downloaded")
        return dest

    print(f"  [fetch] {label}")
    print(f"          {url}")

    headers = {"User-Agent": "SECO-BuildingIntelligence/1.0 (giuseppecardinale7@gmail.com)"}
    resp = requests.get(url, stream=True, timeout=120, headers=headers)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=label[:40], leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))

    size = dest.stat().st_size
    _record(prov, key, dest, url, size)
    print(f"  [ok]   saved {size/1e6:.1f} MB → {dest.name}")
    return dest


# ── 1. Geospatial vector data (BD-L-GeoBase + PAG) ──────────────────────────

def ingest_geobase(prov: dict) -> None:
    print("\n── BD-L-GeoBase (ACT, 2026-04-01) ──")
    _download(BUILDINGS_URL,  BRONZE / "buildings.zip",
              "Buildings (all LU)", prov, "buildings")
    _download(ADDRESSES_URL,  BRONZE / "addresses.zip",
              "Addresses (all LU)", prov, "addresses")
    _download(ADMIN_URL,      BRONZE / "admin_units.zip",
              "Admin units (all LU)", prov, "admin_units")


def ingest_pag(prov: dict) -> None:
    print("\n── National PAG GeoPackage (all communes, 2026-06-15) ──")
    _download(PAG_URL, BRONZE / "pag_national.gpkg.zip",
              "PAG national (266 MB)", prov, "pag_national")


# ── 2. OpenStreetMap — Overpass API ─────────────────────────────────────────

def ingest_osm(prov: dict) -> None:
    print("\n── OpenStreetMap / Overpass — Esch-sur-Alzette buildings ──")
    dest = BRONZE / "osm_esch_buildings.json"
    key = "osm_esch_buildings"

    if _already_fetched(prov, key) and dest.exists():
        print("  [skip] OSM buildings — already downloaded")
        return

    print(f"  [fetch] Overpass QL query (bbox Esch-sur-Alzette)")
    headers = {"User-Agent": "SECO-BuildingIntelligence/1.0"}

    # Try primary endpoint with up to 3 retries; fall back to alternative mirror
    endpoints = [
        OVERPASS_URL,
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for attempt, url in enumerate(endpoints * 2, 1):
        try:
            time.sleep(2 * (attempt - 1))   # back-off: 0s, 2s, 4s, 6s
            resp = requests.post(url,
                                 data={"data": OVERPASS_QUERY},
                                 timeout=120,
                                 headers=headers)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            size = dest.stat().st_size
            _record(prov, key, dest, url, size)
            print(f"  [ok]   saved {size/1e6:.1f} MB → {dest.name}")
            return
        except requests.RequestException as exc:
            print(f"  [warn] Overpass attempt {attempt} failed: {exc}")

    print("  [warn] All Overpass attempts failed — OSM era data will be unavailable")


# ── 3. Regulation HTML (for RAG corpus) ─────────────────────────────────────

def ingest_regulations(prov: dict) -> None:
    print("\n── Regulation HTML pages (for RAG) ──")
    headers = {
        "User-Agent": "SECO-BuildingIntelligence/1.0 (giuseppecardinale7@gmail.com)",
        "Accept-Language": "fr,en;q=0.8",
    }
    regs_meta = []

    for src in REGULATION_SOURCES:
        dest = BRONZE / f"reg_{src['id']}.html"
        key  = f"reg_{src['id']}"

        if _already_fetched(prov, key) and dest.exists():
            print(f"  [skip] {src['label'][:60]}")
            regs_meta.append({"id": src["id"], "path": str(dest), **src})
            continue

        print(f"  [fetch] {src['label'][:60]}")
        try:
            time.sleep(1)   # polite crawl delay
            resp = requests.get(src["url"], timeout=30, headers=headers,
                                allow_redirects=True)
            if resp.status_code == 404:
                print(f"  [warn]  404 — skipping {src['url']}")
                continue
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            size = dest.stat().st_size
            _record(prov, key, dest, src["url"], size)
            print(f"  [ok]   {size/1e3:.0f} KB → {dest.name}")
            regs_meta.append({"id": src["id"], "path": str(dest), **src})
        except requests.RequestException as exc:
            print(f"  [error] {src['id']}: {exc}")

    # Save regulation manifest so silver layer knows what we fetched
    manifest = BRONZE / "regulations_manifest.json"
    manifest.write_text(json.dumps(regs_meta, indent=2, ensure_ascii=False))
    print(f"  [ok]   regulation manifest → {manifest.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BRONZE LAYER — raw data ingestion")
    print(f"Target directory: {BRONZE}")
    print("=" * 60)

    prov = _load_provenance()

    ingest_geobase(prov)
    ingest_pag(prov)
    ingest_osm(prov)
    ingest_regulations(prov)

    print("\n[done] Bronze ingestion complete.")
    print(f"       Provenance log: {PROVENANCE_FILE}")


if __name__ == "__main__":
    main()
