"""
04_enrich.py
------------
Enriches every unique border paper in voids.json with:
  - citation_count  : int   (number of papers citing this one)
  - year            : int   (publication year)
  - abstract        : str   (paper abstract)

Strategy per paper (arxiv ID as DOI):
  1. OpenAlex API   — free, no key, ~10 req/s, has citations + year + abstract
  2. Fall back to arxiv HTML — scrape /abs/{id} for abstract + year
                               (citation_count will be None for these)

OpenAlex accepts arxiv IDs directly:
  https://api.openalex.org/works/arxiv:1810.05718

Abstract comes back as an inverted index (word → [positions]) which we
reconstruct into a plain string.

Input  : voids.json
Output : voids_enriched.json  — same schema as voids.json with three extra
                                 fields added to each border paper:
                                   citation_count, year, abstract, enriched_via

Rate limiting:
  OpenAlex: polite pool is ~10 req/s unauthenticated. We use 0.12s delay.
  arxiv HTML: 1 req/s (polite)

Dependencies:
  pip install requests beautifulsoup4
"""

import json
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT  = "voids.json"
OUTPUT = "voids_enriched.json"

OPENALEX_DELAY  = 0.12   # ~8 req/s, well under their polite limit
ARXIV_DELAY     = 1.05   # 1 req/s for arxiv HTML scraping
REQUEST_TIMEOUT = 15

OPENALEX_BASE = "https://api.openalex.org/works/https://doi.org/10.48550/arXiv.{arxiv_id}"
ARXIV_ABS     = "https://arxiv.org/abs/{arxiv_id}"

HEADERS = {
    "User-Agent": "research-void-enricher/1.0 (academic use; contact via github)",
    # OpenAlex asks for a mailto in the UA for the polite pool (faster, more stable)
    # Swap in your email if you have one:
    # "User-Agent": "research-void-enricher/1.0 (mailto:you@example.com)",
}

EMPTY_ENRICHMENT = {
    "citation_count": None,
    "year":           None,
    "abstract":       None,
    "enriched_via":   None,
}


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------
def reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    OpenAlex stores abstracts as an inverted index:
      { "word": [pos1, pos2, ...], ... }
    Reconstruct into a plain string.
    """
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def openalex_url(arxiv_id: str) -> str:
    """
    Modern IDs (e.g. 2011.10517) → DOI format, better coverage.
    Legacy IDs (e.g. cond-mat/9802200) → arxiv: filter format.
    """
    if "/" in arxiv_id:
        return f"https://api.openalex.org/works/arxiv:{arxiv_id}"
    return OPENALEX_BASE.format(arxiv_id=arxiv_id)


def fetch_openalex(arxiv_id: str) -> dict | None:
    """
    Query OpenAlex for a single arxiv paper.
    Returns enrichment dict or None on failure / missing abstract.
    """
    url = openalex_url(arxiv_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            print(f"    OpenAlex rate limited — waiting 60s ...")
            time.sleep(60)
            return None
        if not resp.ok:
            print(f"    OpenAlex error {resp.status_code} for {arxiv_id}")
            return None

        data = resp.json()

        abstract = reconstruct_abstract(data.get("abstract_inverted_index"))
        year     = data.get("publication_year")
        ccount   = data.get("cited_by_count")

        # Treat missing abstract as a miss — fall back to arxiv HTML
        if not abstract:
            return None

        return {
            "citation_count": ccount,
            "year":           year,
            "abstract":       abstract,
            "enriched_via":   "openalex",
        }

    except Exception as e:
        print(f"    OpenAlex exception for {arxiv_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# arxiv HTML fallback
# ---------------------------------------------------------------------------
def fetch_arxiv_html(arxiv_id: str) -> dict | None:
    """
    Scrape the arxiv abstract page for year and abstract.
    Citation count unavailable this way — returns None for it.
    """
    url = ARXIV_ABS.format(arxiv_id=arxiv_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if not resp.ok:
            print(f"    arxiv HTML error {resp.status_code} for {arxiv_id}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Abstract
        abstract_block = soup.find("blockquote", class_="abstract")
        abstract = ""
        if abstract_block:
            for span in abstract_block.find_all("span", class_="descriptor"):
                span.decompose()
            abstract = abstract_block.get_text(separator=" ").strip()
            abstract = re.sub(r"\s+", " ", abstract)

        # Year — from the submission dateline
        year = None
        dateline = soup.find("div", class_="dateline")
        if dateline:
            m = re.search(r"\b(19|20)\d{2}\b", dateline.get_text())
            if m:
                year = int(m.group())

        if not abstract:
            return None

        return {
            "citation_count": None,
            "year":           year,
            "abstract":       abstract,
            "enriched_via":   "arxiv_html",
        }

    except Exception as e:
        print(f"    arxiv HTML exception for {arxiv_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------
print(f"Reading {INPUT} ...")
with open(INPUT) as f:
    voids: list[dict] = json.load(f)

all_dois: set[str] = set()
for v in voids:
    for p in v.get("border_papers", []):
        doi = p.get("DOI", "").strip()
        if doi and doi != "null":
            all_dois.add(doi)

total_papers = sum(len(v["border_papers"]) for v in voids)
print(f"  {len(voids)} voids, {total_papers} total border paper slots")
print(f"  {len(all_dois)} unique arxiv IDs to enrich\n")

cache: dict[str, dict] = {}
total          = len(all_dois)
success_oa     = 0
success_arxiv  = 0
failed         = 0

for i, arxiv_id in enumerate(sorted(all_dois)):
    print(f"  [{i+1:>4}/{total}] {arxiv_id:<30}", end=" ", flush=True)

    # --- Primary: OpenAlex ---
    result = fetch_openalex(arxiv_id)
    time.sleep(OPENALEX_DELAY)

    if result:
        print(f"✓ OA   | year={result['year']} cites={result['citation_count']}")
        cache[arxiv_id] = result
        success_oa += 1
        continue

    # --- Fallback: arxiv HTML ---
    print(f"→ arxiv", end=" ", flush=True)
    result = fetch_arxiv_html(arxiv_id)
    time.sleep(ARXIV_DELAY)

    if result:
        print(f"✓ HTML | year={result['year']}")
        cache[arxiv_id] = result
        success_arxiv += 1
        continue

    # --- Give up ---
    print("✗ failed")
    cache[arxiv_id] = {**EMPTY_ENRICHMENT}
    failed += 1

print(f"\nEnrichment complete:")
print(f"  OpenAlex   : {success_oa}")
print(f"  arxiv HTML : {success_arxiv}")
print(f"  Failed     : {failed}")

# ---------------------------------------------------------------------------
# Inject enrichment back into voids structure
# ---------------------------------------------------------------------------
for v in voids:
    for p in v.get("border_papers", []):
        doi = p.get("DOI", "").strip()
        enrichment = cache.get(doi, {**EMPTY_ENRICHMENT}) if (doi and doi != "null") else {**EMPTY_ENRICHMENT}
        p["citation_count"] = enrichment["citation_count"]
        p["year"]           = enrichment["year"]
        p["abstract"]       = enrichment["abstract"]
        p["enriched_via"]   = enrichment["enriched_via"]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT, "w") as f:
    json.dump(voids, f, indent=2)

print(f"\nSaved → {OUTPUT}")

# Quality check
abstracts_filled = sum(1 for v in voids for p in v["border_papers"] if p.get("abstract"))
total_slots      = sum(len(v["border_papers"]) for v in voids)
oa_count         = sum(1 for v in voids for p in v["border_papers"] if p.get("enriched_via") == "openalex")
html_count       = sum(1 for v in voids for p in v["border_papers"] if p.get("enriched_via") == "arxiv_html")

print(f"  Total slots     : {total_slots}")
print(f"  Abstract fill   : {abstracts_filled}/{total_slots} ({100*abstracts_filled/max(total_slots,1):.1f}%)")
print(f"  via OpenAlex    : {oa_count}")
print(f"  via arxiv HTML  : {html_count}")