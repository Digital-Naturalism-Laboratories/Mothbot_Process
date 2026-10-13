"""Identify a GBIF species-list CSV: its DOI, citation reference and taxon-ID system.

GBIF species-list downloads come in two styles:

* Old: renamed by hand to carry the DOI, e.g.
  ``SpeciesList_CountryPanama_TaxaInsecta_doi.org10.15468dl.6nxkw6.csv``.
  Taxon keys are GBIF Backbone integers (``1090772``).
* New (Oct 2026+): GBIF's own file name is the download key, e.g.
  ``0024609-260928105237408.csv``. Taxonomy comes from the Catalogue of Life
  (COL XR) and taxon keys are alphanumeric (``B6L67``, ``BOLD.AAA1580``).

For a download-key name the DOI isn't in the file at all; it comes from the
GBIF API. Everything here works offline: when GBIF can't be reached the list
is referenced by its GBIF download page URL (which resolves to the DOI later)
and the DOI is reported as pending. Once fetched, the API response is saved
next to the CSV as ``<csv stem>.gbif.json`` so later runs — and Mothbot
Classify, which reads the same file — have it offline.
"""

import csv
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

GBIF_DOWNLOAD_API = "https://api.gbif.org/v1/occurrence/download/"
GBIF_DOWNLOAD_PAGE = "https://www.gbif.org/occurrence/download/"
COL_XR_CHECKLIST_KEY = "7ddf754f-d193-4cc9-b351-99906754a03b"

SIDECAR_SUFFIX = ".gbif.json"

# Taxon-ID systems, shared with Mothbot Classify.
TAXON_ID_GBIF_BACKBONE = "gbif-backbone"
TAXON_ID_COL = "col"
TAXON_ID_LABELS = {
    TAXON_ID_GBIF_BACKBONE: "GBIF Backbone Taxonomy",
    TAXON_ID_COL: "Catalogue of Life (COL XR, via GBIF)",
}

_DOWNLOAD_KEY_RE = re.compile(r"(?<!\d)(\d{7}-\d{15})(?!\d)")

_COUNTRY_NAMES = {
    "PA": "Panama", "CR": "Costa Rica", "US": "USA", "EC": "Ecuador", "CO": "Colombia",
    "PE": "Peru", "BR": "Brazil", "MX": "Mexico", "ID": "Indonesia", "GB": "United Kingdom",
    "CA": "Canada", "AU": "Australia", "IN": "India", "CN": "China", "DE": "Germany",
}
_TAXON_NAMES = {"H6": "Insecta", "216": "Insecta", "797": "Lepidoptera", "RT": "Arthropoda", "54": "Arthropoda"}


def download_key_from_filename(csv_path):
    """GBIF download key (``0024609-260928105237408``) in the file name, or None."""
    m = _DOWNLOAD_KEY_RE.search(os.path.basename(str(csv_path)))
    return m.group(1) if m else None


def doi_from_filename(csv_path):
    """DOI URL from a hand-renamed file name, or None.

    Handles ``…_doi.org10.15468dl.epzeza.csv`` (no separator dot) and
    ``…_doi.org.10.15468.dl.pkbmuj.csv`` (dot-separated).
    """
    filename = os.path.basename(str(csv_path))
    match = re.search(r"(doi\.org[0-9A-Za-z\.\-]+)", filename)
    if not match:
        return None
    doi_core = match.group(1).replace("doi.org", "").lstrip(".")
    if doi_core.lower().endswith(".csv"):
        doi_core = doi_core[:-4]
    # DOI prefix is always "10.<registrant>" — find the split point.
    m = re.match(r"(10\.\d+)(.+)", doi_core)
    if not m:
        return None
    prefix, suffix = m.groups()
    return f"https://doi.org/{prefix}/{suffix.lstrip('.')}"


def doi_url(doi):
    """``10.15468/dl.x3qf72`` → ``https://doi.org/10.15468/dl.x3qf72``."""
    doi = (doi or "").strip()
    if not doi:
        return None
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.I).removeprefix("doi:")
    return f"https://doi.org/{doi}"


def sidecar_path(csv_path):
    return os.path.splitext(str(csv_path))[0] + SIDECAR_SUFFIX


def load_sidecar(csv_path):
    """Saved GBIF download metadata for this CSV, or None."""
    path = sidecar_path(csv_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data.get("doi") else None
    except (OSError, ValueError):
        return None


def fetch_gbif_download(key_or_doi, timeout=6.0):
    """GBIF download metadata (dict) for a download key or DOI, or None if unreachable."""
    ref = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", str(key_or_doi).strip())
    try:
        req = urllib.request.Request(GBIF_DOWNLOAD_API + ref, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, dict) and data.get("doi") else None
    except Exception:
        return None


def describe_predicate(predicate):
    """Short human description of a GBIF download filter, e.g. "Insecta · Panama, Costa Rica"."""
    taxa, countries = [], []

    def walk(p):
        if not isinstance(p, dict):
            return
        for sub in p.get("predicates") or []:
            walk(sub)
        if p.get("predicate"):
            walk(p["predicate"])
        key = p.get("key")
        values = p.get("values") or ([p["value"]] if "value" in p else [])
        if key == "TAXON_KEY":
            taxa.extend(_TAXON_NAMES.get(str(v), f"taxon {v}") for v in values)
        elif key == "COUNTRY":
            countries.extend(_COUNTRY_NAMES.get(str(v).upper(), str(v).upper()) for v in values)

    walk(predicate)
    parts = [", ".join(dict.fromkeys(taxa)) or None, ", ".join(dict.fromkeys(countries)) or "Worldwide"]
    return " · ".join(p for p in parts if p)


def _sidecar_from_api(data):
    request = data.get("request") or {}
    checklist = request.get("checklistKey")
    return {
        "source": "gbif-download",
        "downloadKey": data.get("key"),
        "doi": data.get("doi"),
        "doiUrl": doi_url(data.get("doi")),
        "license": data.get("license"),
        "created": data.get("created"),
        "totalRecords": data.get("totalRecords"),
        "checklistKey": checklist,
        "checklist": "Catalogue of Life (COL XR)" if checklist == COL_XR_CHECKLIST_KEY
                     else (checklist or "GBIF Backbone Taxonomy"),
        "description": describe_predicate(request.get("predicate")),
        "predicate": request.get("predicate"),
        "fetchedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_sidecar(csv_path, info):
    """Write the sidecar next to the CSV; returns False (silently) if the folder is read-only."""
    try:
        with open(sidecar_path(csv_path), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return True
    except OSError:
        return False


def detect_taxon_id_system(csv_path, sample_rows=2000):
    """Which ID system the list's taxonKey column uses: TAXON_ID_COL, TAXON_ID_GBIF_BACKBONE or None.

    Backbone keys are always integers; COL XR keys are alphanumeric. Looks at
    the first ``sample_rows`` rows, so it's cheap even on the worldwide list.
    """
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames or "taxonKey" not in reader.fieldnames:
                return None
            numeric = alpha = 0
            for i, row in enumerate(reader):
                if i >= sample_rows:
                    break
                key = (row.get("taxonKey") or "").strip()
                if not key:
                    continue
                if key.isdigit():
                    numeric += 1
                else:
                    alpha += 1
    except OSError:
        return None
    if not numeric and not alpha:
        return None
    return TAXON_ID_COL if alpha > numeric else TAXON_ID_GBIF_BACKBONE


def resolve_species_list(csv_path, allow_network=True, timeout=6.0):
    """Everything needed to cite a species list. Never raises; works offline.

    Returns a dict:
      ref          — what to store in each detection's ``species_list``: the DOI
                     URL when known, else the GBIF download page URL, else "no_doi"
      doi_url      — DOI URL or None
      doi_status   — "resolved" (from GBIF / sidecar), "filename" (hand-renamed file),
                     "pending" (download key known, GBIF not reachable yet) or "none"
      download_key — GBIF download key or None
      description  — e.g. "Insecta · Panama, Costa Rica" (when known)
      taxon_id_system — TAXON_ID_COL / TAXON_ID_GBIF_BACKBONE / None
      message      — one line for the log
    """
    info = {
        "ref": "no_doi", "doi_url": None, "doi_status": "none", "download_key": None,
        "description": None, "taxon_id_system": None, "message": "",
    }
    if not csv_path:
        info["message"] = "No species list — IDs won't reference one."
        return info

    info["taxon_id_system"] = detect_taxon_id_system(csv_path)
    key = download_key_from_filename(csv_path)
    info["download_key"] = key

    sidecar = load_sidecar(csv_path)
    if sidecar is None and allow_network and (key or doi_from_filename(csv_path)):
        data = fetch_gbif_download(key or doi_from_filename(csv_path), timeout=timeout)
        if data:
            sidecar = _sidecar_from_api(data)
            save_sidecar(csv_path, sidecar)

    if sidecar:
        info.update(
            ref=sidecar.get("doiUrl") or doi_url(sidecar.get("doi")),
            doi_url=sidecar.get("doiUrl") or doi_url(sidecar.get("doi")),
            doi_status="resolved",
            download_key=sidecar.get("downloadKey") or key,
            description=sidecar.get("description"),
        )
        info["message"] = f"Species list DOI: {info['doi_url']}" + (
            f" ({info['description']})" if info["description"] else "")
        return info

    filename_doi = doi_from_filename(csv_path)
    if filename_doi:
        info.update(ref=filename_doi, doi_url=filename_doi, doi_status="filename")
        info["message"] = f"Species list DOI (from file name): {filename_doi}"
        return info

    if key:
        info.update(ref=GBIF_DOWNLOAD_PAGE + key, doi_status="pending")
        info["message"] = (
            f"⚠️  DOI not retrieved yet for GBIF download {key} (GBIF unreachable — offline?). "
            f"Continuing; IDs will reference {info['ref']} until a run can reach GBIF."
        )
        return info

    info["message"] = (
        "⚠️  No DOI for this species list: its file name has neither a GBIF download key "
        "nor a doi.org… DOI. IDs will be stored with species_list = \"no_doi\"."
    )
    return info


def species_list_aliases(info):
    """All references that mean this same list (DOI URL and GBIF download page URL)."""
    aliases = {info.get("ref")}
    if info.get("doi_url"):
        aliases.add(info["doi_url"])
    if info.get("download_key"):
        aliases.add(GBIF_DOWNLOAD_PAGE + info["download_key"])
    aliases.discard(None)
    return aliases
