"""
Step 1 — download raw data from data.public.lu (Luxembourg open data).

Run once. Already-downloaded files are skipped automatically.
"""

import requests
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

# Two public datasets, both from data.public.lu
SOURCES = [
    {
        "name": "BD-L-GeoBase buildings (ACT, April 2026)",
        "url": "https://download.data.public.lu/resources/bd-l-geobase/20260401-011245/buildings-20260401.zip",
        "dest": DATA / "buildings.zip",
    },
    {
        "name": "National PAG zoning plan (June 2026, 266 MB)",
        "url": "https://download.data.public.lu/resources/pag-geometries-de-tous-les-pag-version-2011-en-vigueur/20260615-023539/pag.gpkg.zip",
        "dest": DATA / "pag.zip",
    },
]

for src in SOURCES:
    dest = src["dest"]
    if dest.exists():
        print(f"  skip  {src['name']} (already downloaded)")
        continue

    print(f"  downloading  {src['name']} ...")
    r = requests.get(src["url"], stream=True, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  saved  {dest.stat().st_size / 1e6:.0f} MB  →  {dest.name}")

print("\nDone. Run 02_prepare.py next.")
