"""
05_rank.py
----------
For each void, selects the best 8 border papers from the enriched candidate
pool using a multi-metric scoring approach optimised for cross-pollination:
generating genuinely novel ideas by bridging the gap between surrounding fields.

Metrics (in order of priority):
  1. Angular diversity   — geometric spread around void centroid in UMAP space.
                           Ensures papers surround the void from all directions.
                           Implemented via angular sector bucketing (8 sectors).

  2. Vector diversity    — pairwise cosine dissimilarity between selected papers
                           in the original 768-dim embedding space. Prevents
                           picking papers that are semantically near-duplicates
                           even if they appear in different UMAP positions.
                           Requires join back to the full parquet for vectors.

  3. Cluster diversity   — at most MAX_PER_CLUSTER papers from the same cluster.
                           Soft constraint applied after angular bucketing.

  4. Citation count      — log-scaled. Higher cited = more load-bearing to its
                           field. Used as tiebreaker within a sector/cluster slot.
                           None → treated as 0.

  5. Recency             — papers closer to CURRENT_YEAR score higher. Represents
                           the leading edge of a field rather than its history.
                           None → treated as neutral (CURRENT_YEAR - 5).

Algorithm:
  Phase 1 — Angular bucketing:
    Divide 360° around the void centroid into N_SELECT sectors.
    For each occupied sector, score candidates by:
      citation_score * WEIGHT_CITATION + recency_score * WEIGHT_RECENCY
    Pick the best candidate per sector. This guarantees geometric spread.

  Phase 2 — Fill remaining slots:
    If fewer than N_SELECT sectors were occupied, fill remaining slots from
    unused candidates ranked by combined score, subject to:
      - cluster cap (MAX_PER_CLUSTER)
      - minimum vector distance from already-selected papers (MIN_VECTOR_SIM)

  Phase 3 — Output:
    Each selected paper gets a breakdown of its component scores.

Input  : voids_enriched.json   (from 04_enrich.py)
         umap_200k.parquet     (for cluster membership, already in border papers)
         sample_200k.parquet   (NOT used — vectors not saved here)
         [full parquet path]   (for 768-dim vectors, joined by DOI)

Output : voids_ranked.json     — voids_enriched.json schema + per-void
                                  "selected_papers" list of 8

Dependencies:
  pip install pandas numpy pyarrow scikit-learn
"""

import json
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_VOIDS   = "voids_enriched.json"
FULL_PARQUET  = "sample_200k.parquet"   # has vector column; change to full
                                         # 2.25M parquet path if preferred
OUTPUT        = "voids_ranked.json"

N_SELECT        = 8    # papers to select per void
MAX_PER_CLUSTER = 2    # max papers from same cluster in final selection
MIN_VECTOR_SIM  = 0.85 # cosine similarity threshold — pairs above this are
                        # considered near-duplicates; second one penalised
CURRENT_YEAR    = 2024
FALLBACK_YEAR   = CURRENT_YEAR - 5   # used when year is None

# Score weights for within-sector/slot tiebreaking
WEIGHT_CITATION = 0.5
WEIGHT_RECENCY  = 0.5

# ---------------------------------------------------------------------------
# Load voids
# ---------------------------------------------------------------------------
print(f"Reading {INPUT_VOIDS} ...")
with open(INPUT_VOIDS) as f:
    voids: list[dict] = json.load(f)
print(f"  {len(voids)} voids loaded")

# ---------------------------------------------------------------------------
# Load vectors from parquet (join by DOI)
# ---------------------------------------------------------------------------
print(f"\nLoading vectors from {FULL_PARQUET} ...")
pf = pq.ParquetFile(FULL_PARQUET)
df = pf.read(columns=["DOI", "vector"]).to_pandas()

# Build DOI → vector lookup
print(f"  {len(df):,} rows — building DOI → vector index ...")
doi_to_vector: dict[str, np.ndarray] = {}
for _, row in df.iterrows():
    doi = str(row["DOI"]).strip()
    vec = row["vector"]
    if vec is not None and doi and doi != "null":
        doi_to_vector[doi] = np.asarray(vec, dtype=np.float32)

del df  # free memory
print(f"  {len(doi_to_vector):,} vectors indexed")

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def citation_score(count: int | None) -> float:
    """Log-normalised citation count → [0, 1] proxy. Uncited = 0."""
    if count is None or count <= 0:
        return 0.0
    return math.log1p(count) / math.log1p(50_000)   # ~50k as soft ceiling


def recency_score(year: int | None) -> float:
    """Linear recency → [0, 1]. Papers from CURRENT_YEAR = 1.0, 20yr old = 0."""
    y = year if year is not None else FALLBACK_YEAR
    age = max(0, CURRENT_YEAR - y)
    return max(0.0, 1.0 - age / 20.0)


def combined_score(p: dict) -> float:
    return (
        WEIGHT_CITATION * citation_score(p.get("citation_count")) +
        WEIGHT_RECENCY  * recency_score(p.get("year"))
    )


def angle_to_centroid(cx: float, cy: float, px: float, py: float) -> float:
    """Angle in degrees [0, 360) from void centroid to paper position."""
    return math.degrees(math.atan2(py - cy, px - cx)) % 360


def sector_of(angle: float, n_sectors: int = N_SELECT) -> int:
    sector_size = 360.0 / n_sectors
    return int(angle / sector_size) % n_sectors


def vectors_of(papers: list[dict]) -> np.ndarray | None:
    """Stack 768-dim vectors for a list of papers. Returns None if none found."""
    vecs = []
    for p in papers:
        doi = p.get("DOI", "").strip()
        if doi in doi_to_vector:
            vecs.append(doi_to_vector[doi])
        else:
            vecs.append(None)
    return vecs  # list of np.ndarray | None, parallel to papers


def is_vector_duplicate(
    vec: np.ndarray | None,
    selected_vecs: list[np.ndarray],
    threshold: float = MIN_VECTOR_SIM,
) -> bool:
    """True if vec is too similar to any already-selected vector."""
    if vec is None or not selected_vecs:
        return False
    sims = cosine_similarity([vec], selected_vecs)[0]
    return bool(sims.max() >= threshold)


# ---------------------------------------------------------------------------
# Per-void ranking
# ---------------------------------------------------------------------------
print(f"\nRanking border papers — selecting {N_SELECT} per void ...")

for v in voids:
    cx, cy    = v["centroid"]
    candidates = v.get("border_papers", [])

    if not candidates:
        v["selected_papers"] = []
        continue

    # Precompute per-candidate derived fields
    for p in candidates:
        p["_angle"]    = angle_to_centroid(cx, cy, p["x"], p["y"])
        p["_sector"]   = sector_of(p["_angle"])
        p["_score"]    = combined_score(p)
        p["_vec"]      = doi_to_vector.get(p.get("DOI", "").strip())

    # ── Phase 1: Angular bucketing ───────────────────────────────────────────
    # Group candidates by sector, pick best-scoring from each
    sectors: dict[int, list[dict]] = {}
    for p in candidates:
        sectors.setdefault(p["_sector"], []).append(p)

    selected:       list[dict]      = []
    selected_vecs:  list[np.ndarray] = []
    cluster_counts: dict[int, int]  = {}
    used_dois:      set[str]        = set()

    for sector_id in sorted(sectors.keys()):
        if len(selected) >= N_SELECT:
            break

        # Sort sector candidates by score descending
        bucket = sorted(sectors[sector_id], key=lambda p: p["_score"], reverse=True)

        for p in bucket:
            doi = p.get("DOI", "").strip()
            if doi in used_dois:
                continue
            cluster = p.get("cluster", -1)
            if cluster_counts.get(cluster, 0) >= MAX_PER_CLUSTER:
                continue
            if is_vector_duplicate(p["_vec"], selected_vecs):
                continue

            # Accept
            selected.append(p)
            if p["_vec"] is not None:
                selected_vecs.append(p["_vec"])
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            used_dois.add(doi)
            break   # one per sector in phase 1

    # ── Phase 2: Fill remaining slots ───────────────────────────────────────
    if len(selected) < N_SELECT:
        remaining = sorted(
            [p for p in candidates if p.get("DOI", "") not in used_dois],
            key=lambda p: p["_score"],
            reverse=True,
        )
        for p in remaining:
            if len(selected) >= N_SELECT:
                break
            doi     = p.get("DOI", "").strip()
            cluster = p.get("cluster", -1)
            if cluster_counts.get(cluster, 0) >= MAX_PER_CLUSTER:
                continue
            if is_vector_duplicate(p["_vec"], selected_vecs):
                continue

            selected.append(p)
            if p["_vec"] is not None:
                selected_vecs.append(p["_vec"])
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            used_dois.add(doi)

    # ── Build output ─────────────────────────────────────────────────────────
    selected_out = []
    for rank, p in enumerate(selected):
        selected_out.append({
            "rank":           rank,
            "title":          p["title"],
            "DOI":            p["DOI"],
            "x":              p["x"],
            "y":              p["y"],
            "cluster":        p["cluster"],
            "citation_count": p.get("citation_count"),
            "year":           p.get("year"),
            "abstract":       p.get("abstract"),
            "enriched_via":   p.get("enriched_via"),
            "scores": {
                "combined":  round(p["_score"], 4),
                "citation":  round(citation_score(p.get("citation_count")), 4),
                "recency":   round(recency_score(p.get("year")), 4),
                "sector":    p["_sector"],
                "angle_deg": round(p["_angle"], 2),
            },
        })

    v["selected_papers"] = selected_out

    # Clean up temp keys
    for p in candidates:
        for k in ("_angle", "_sector", "_score", "_vec"):
            p.pop(k, None)

    n_clusters_repr = len({p["cluster"] for p in selected})
    print(f"  Void {v['void_id']:>2} \"{v['name'][:40]}\"")
    print(f"         selected {len(selected_out)}/{len(candidates)} papers "
          f"· {n_clusters_repr} clusters · "
          f"sectors: {sorted({p['scores']['sector'] for p in selected_out})}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT, "w") as f:
    json.dump(voids, f, indent=2)

print(f"\n{'='*60}")
print(f"Saved → {OUTPUT}")

# Summary stats
total_selected = sum(len(v["selected_papers"]) for v in voids)
avg_clusters   = np.mean([
    len({p["cluster"] for p in v["selected_papers"]})
    for v in voids if v["selected_papers"]
])
print(f"Total selected papers : {total_selected}")
print(f"Avg clusters per void : {avg_clusters:.1f}")
print(f"\nTop 3 voids:")
for v in voids[:3]:
    print(f'\n  "{v["name"]}"')
    for p in v["selected_papers"]:
        print(f'    [{p["rank"]}] {p["title"][:60]}'
              f' | {p["year"]} | {p["citation_count"]} cites'
              f' | score={p["scores"]["combined"]:.3f}'
              f' | sector={p["scores"]["sector"]}')