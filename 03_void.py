"""
03_voids.py
-----------
Detect low-density void regions in 2D UMAP space, find the border papers
that ring each void, compute the void's convex-hull shape, and name each
void using an LLM — "what topic is conspicuously absent here?"

Pipeline:
  1. Compute Voronoi diagram of all paper (x, y) coordinates
  2. Filter Voronoi vertices: must be inside alpha-shape hull
  3. Score each vertex by distance to nearest real paper
     (Voronoi vertices ARE the centres of maximal empty circles — no KDE needed)
  3b. Filter by angular coverage: discard edge bays / peninsulas where border
      papers don't surround the candidate from all sides
  4. Deduplicate: suppress vertices too close to a higher-ranked one
  5. Pick top N → void centres
  6. Per void: find K nearest real papers → border papers
  7. Per void: convex hull of border paper coords → shape + area
  8. Per void: send border titles to LLM → topic name  [currently disabled]

Why Voronoi instead of KDE grid:
  Voronoi vertices are by construction the maximally-empty points — each one
  is the centre of the largest empty circle that fits between its neighbours.
  This is the exact answer to "where are the holes?" with no grid resolution
  or bandwidth hyperparameters, and no edge bias from density estimation.

Why angular coverage filtering:
  The alpha-shape hull clips the outer boundary of the data, but within that
  hull there can be long thin peninsulas or concave bays where the hull still
  "covers" a huge stretch of empty space that's really just open edge, not a
  true interior gap. A coastal bay has all its border papers clustered on one
  arc (~180°), with the other half pointing into empty space. A real interior
  void has papers surrounding it from all directions. The angular coverage
  filter directly measures the largest angular gap between consecutive border
  papers around each candidate — large gap = edge artifact, discard it.

LLM backend : HackClub AI proxy via openrouter
  server_url : https://ai.hackclub.com/proxy/v1
  model      : openai/gpt-oss-120b
  API key    : HACKCLUB_KEY in .env

Input  : umap_200k.parquet   (columns: id, title, DOI, x, y, cluster)
Output : voids.json

Output schema per void:
  {
    "void_id":        0,
    "void_rank":      0,
    "centroid":       [x, y],
    "empty_radius":   1.234,               // radius of largest empty circle (UMAP units)
    "name":           "Quantum-Biological Energy Transduction",
    "name_reasoning": "...",
    "shape": {
      "type":     "convex_hull",
      "vertices": [[x0,y0], [x1,y1], ...]  // ordered CCW, open polygon
    },
    "shape_area":    1.234,                // UMAP units²
    "border_papers": [
      {"title": "...", "DOI": "...", "x": ..., "y": ..., "cluster": ...},
      ...
    ]
  }

Dependencies:
  pip install pandas numpy scikit-learn alphashape shapely joblib pyarrow python-dotenv scipy
  echo "HACKCLUB_KEY=your_key_here" > .env
"""

import json
import os
import time
import multiprocessing

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from shapely.geometry import MultiPoint, Point
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi
import alphashape

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT  = "umap_200k.parquet"
OUTPUT = "voids.json"

N_VOIDS = 50

# Alpha shape (concave hull of entire corpus)
# Lower = looser hull (more candidates survive), higher = tighter
ALPHA = 0.05

# Void deduplication: suppress a vertex if a higher-ranked one is within
# this distance. Prevents N voids from all clustering in one big hole.
# Rule of thumb: ~5–10% of the data's bounding-box diagonal.
DEDUP_RADIUS = 1.0

# Border papers per void
BORDER_K = 10

# Angular coverage filter
# How many neighbors to use when computing angular spread around each candidate.
# More neighbors = more robust estimate, but slower.
ANGULAR_K = 20
# Require that border papers cover at least this many degrees around the candidate.
# Equivalently: the largest angular gap between consecutive neighbors must be
# no more than (360 - MIN_COVERAGE_DEG). Set lower (e.g. 260) if you find
# real interior voids near elongated clusters are being incorrectly discarded.
MIN_COVERAGE_DEG = 300  # allows at most a 60° open wedge

# Parallelism
N_JOBS = -1

# LLM naming — disabled for now
DO_LLM_NAMING = False
LLM_DELAY     = 0.5

# ---------------------------------------------------------------------------
# LLM: name a void given its border paper titles
# ---------------------------------------------------------------------------
def name_void(border_titles: list, void_id: int) -> dict:
    """Call the LLM and return {"name": ..., "reasoning": ...}."""
    # --- LLM naming disabled: return placeholder ---
    return {"name": f"Void {void_id}", "reasoning": ""}


# ---------------------------------------------------------------------------
# Shape: convex hull of border paper coordinates
# ---------------------------------------------------------------------------
def compute_void_shape(border_xy: np.ndarray) -> tuple:
    """
    Compute the convex hull of the K border-paper (x, y) coordinates.

    Returns
    -------
    shape_dict : {"type": "convex_hull", "vertices": [[x,y], ...]}
                 Vertices ordered CCW, open polygon (no repeated closing point).
    area       : float  area in UMAP coordinate units²
    """
    mp   = MultiPoint(border_xy)
    hull = mp.convex_hull

    if hull.geom_type == "Polygon":
        coords = list(hull.exterior.coords)[:-1]
        area   = hull.area
    elif hull.geom_type == "LineString":
        coords = list(hull.coords)
        area   = 0.0
    else:
        coords = border_xy.tolist()
        area   = 0.0

    return (
        {"type": "convex_hull", "vertices": [[float(x), float(y)] for x, y in coords]},
        float(area),
    )


# ---------------------------------------------------------------------------
# Angular coverage helper
# ---------------------------------------------------------------------------
def is_surrounded(centre: np.ndarray, neighbor_xy: np.ndarray, min_arc_deg: float) -> bool:
    """
    Return True if neighbor_xy surrounds centre with no single angular gap
    larger than (360 - min_arc_deg) degrees.

    A candidate on the edge of the data cloud (a coastal bay) will have all
    its neighbors clustered on one side, leaving a huge gap pointing outward —
    this function returns False for it.  A genuine interior void will have
    neighbors spread around it and return True.
    """
    angles = np.degrees(np.arctan2(
        neighbor_xy[:, 1] - centre[1],
        neighbor_xy[:, 0] - centre[0],
    ))
    angles = np.sort(angles % 360)
    # Gaps between consecutive neighbor angles, wrapping around 360→0
    gaps = np.diff(np.append(angles, angles[0] + 360))
    largest_gap = gaps.max()
    return largest_gap <= (360.0 - min_arc_deg)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Reading {INPUT} ...")
df = pd.read_parquet(INPUT)
print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

xy = df[["x", "y"]].values.astype(np.float64)
print(f"  x ∈ [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]")
print(f"  y ∈ [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")

bbox_diag = np.sqrt(
    (xy[:, 0].max() - xy[:, 0].min()) ** 2 +
    (xy[:, 1].max() - xy[:, 1].min()) ** 2
)
print(f"  bounding-box diagonal: {bbox_diag:.2f}  (DEDUP_RADIUS={DEDUP_RADIUS})")

# ---------------------------------------------------------------------------
# Step 1: Voronoi diagram
# ---------------------------------------------------------------------------
print(f"\nComputing Voronoi diagram for {len(xy):,} points ...")
t0  = time.time()
vor = Voronoi(xy)
# vor.vertices shape: (M, 2) — the candidate void centres
voronoi_verts = vor.vertices
print(f"  {len(voronoi_verts):,} Voronoi vertices  ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Step 2: Clip to alpha-shape hull
# ---------------------------------------------------------------------------
print(f"\nComputing alpha shape (alpha={ALPHA}) ...")
t0   = time.time()
hull = alphashape.alphashape(xy.tolist(), ALPHA)
print(f"  Hull type : {hull.geom_type}  ({time.time()-t0:.1f}s)")

print(f"Clipping {len(voronoi_verts):,} Voronoi vertices to hull (n_jobs={N_JOBS}) ...")
t0 = time.time()

def _inside(p):
    return hull.contains(Point(p))

inside_mask = np.array(
    Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_inside)(p) for p in voronoi_verts
    ),
    dtype=bool,
)
candidates  = voronoi_verts[inside_mask]
print(f"  {len(candidates):,} vertices inside hull  ({time.time()-t0:.1f}s)")

if len(candidates) == 0:
    raise RuntimeError(
        "No Voronoi vertices survived the hull clip. "
        "Try lowering ALPHA (e.g. 0.02)."
    )

# ---------------------------------------------------------------------------
# Step 3: Score by distance to nearest real paper
#
# Each Voronoi vertex is the circumcentre of a Delaunay triangle, so its
# distance to the nearest data point IS the radius of the largest empty
# circle centred there. Larger radius = bigger hole = better void candidate.
# ---------------------------------------------------------------------------
print(f"\nScoring {len(candidates):,} candidates by empty-circle radius ...")
t0 = time.time()
nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=N_JOBS).fit(xy)
empty_radii, _ = nbrs.kneighbors(candidates)
empty_radii    = empty_radii[:, 0]
print(f"  radius ∈ [{empty_radii.min():.4f}, {empty_radii.max():.4f}]  ({time.time()-t0:.1f}s)")

# Sort descending: largest empty circle first
order       = np.argsort(-empty_radii)
candidates  = candidates[order]
empty_radii = empty_radii[order]

# ---------------------------------------------------------------------------
# Step 3b: Filter by angular coverage
#
# Discard candidates whose ANGULAR_K nearest papers don't surround them on
# all sides. Edge bays / peninsulas fail this because their neighbors exist
# on only ~half the circle, leaving a giant open-wedge pointing outward.
#
# Tune MIN_COVERAGE_DEG:
#   300° → strict, at most a 60° open wedge allowed (default, good start)
#   270° → looser, allows a 90° gap (use if real voids near filaments are lost)
#   260° → even looser
# ---------------------------------------------------------------------------
print(f"\nFiltering by angular coverage "
      f"(k={ANGULAR_K}, min_coverage={MIN_COVERAGE_DEG}°, "
      f"max_gap={360-MIN_COVERAGE_DEG}°) ...")
t0 = time.time()

angular_nbrs = NearestNeighbors(n_neighbors=ANGULAR_K, n_jobs=N_JOBS).fit(xy)
_, angular_idxs = angular_nbrs.kneighbors(candidates)

surrounded_mask = np.array([
    is_surrounded(c, xy[idxs], MIN_COVERAGE_DEG)
    for c, idxs in zip(candidates, angular_idxs)
])

n_before = len(candidates)
candidates  = candidates[surrounded_mask]
empty_radii = empty_radii[surrounded_mask]
print(f"  {surrounded_mask.sum():,} / {n_before:,} candidates passed  ({time.time()-t0:.1f}s)")

if len(candidates) == 0:
    raise RuntimeError(
        "No candidates survived the angular coverage filter. "
        f"Lower MIN_COVERAGE_DEG (currently {MIN_COVERAGE_DEG}) — try 270 or 260."
    )

# ---------------------------------------------------------------------------
# Step 4: Deduplicate — greedy suppression within DEDUP_RADIUS
#
# Walk the ranked list; accept a candidate only if it is farther than
# DEDUP_RADIUS from every already-accepted void. This prevents N voids
# from all piling into the single largest hole.
# ---------------------------------------------------------------------------
print(f"\nDeduplicating (DEDUP_RADIUS={DEDUP_RADIUS}) ...")
accepted_centres = []
accepted_radii   = []

for pt, r in zip(candidates, empty_radii):
    if len(accepted_centres) == 0:
        accepted_centres.append(pt)
        accepted_radii.append(r)
        continue
    dists = np.linalg.norm(np.array(accepted_centres) - pt, axis=1)
    if dists.min() >= DEDUP_RADIUS:
        accepted_centres.append(pt)
        accepted_radii.append(r)
    if len(accepted_centres) >= N_VOIDS:
        break

void_centres = np.array(accepted_centres)
void_radii   = np.array(accepted_radii)
print(f"  {len(void_centres)} voids after deduplication")

# ---------------------------------------------------------------------------
# Step 5: Border papers — single batched kneighbors call
# ---------------------------------------------------------------------------
print(f"\nFinding {BORDER_K} border papers per void ...")
t0 = time.time()
paper_nbrs = NearestNeighbors(n_neighbors=BORDER_K, algorithm="auto", n_jobs=N_JOBS).fit(xy)
_, border_indices = paper_nbrs.kneighbors(void_centres)
print(f"  Done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Steps 6 & 7: Shape + naming, assemble output
# ---------------------------------------------------------------------------
print(f"\nComputing shapes + skipping LLM naming (DO_LLM_NAMING=False) ...")

voids_output = []

for rank, (centre, radius, b_idxs) in enumerate(
    zip(void_centres, void_radii, border_indices)
):
    # --- border papers ---
    border_papers = []
    border_xy     = []
    border_titles = []

    for idx in b_idxs:
        row = df.iloc[idx]
        border_papers.append({
            "title":   str(row["title"]),
            "DOI":     str(row["DOI"])     if "DOI"     in df.columns else "",
            "x":       float(row["x"]),
            "y":       float(row["y"]),
            "cluster": int(row["cluster"]) if "cluster" in df.columns else -1,
        })
        border_xy.append([float(row["x"]), float(row["y"])])
        border_titles.append(str(row["title"]))

    # --- void shape ---
    shape_dict, shape_area = compute_void_shape(np.array(border_xy))

    # --- naming ---
    naming = name_void(border_titles, rank)

    voids_output.append({
        "void_id":        rank,
        "void_rank":      rank,
        "centroid":       [float(centre[0]), float(centre[1])],
        "empty_radius":   float(radius),
        "name":           naming["name"],
        "name_reasoning": naming["reasoning"],
        "shape": {
            "type":     shape_dict["type"],
            "vertices": shape_dict["vertices"],
        },
        "shape_area":    shape_area,
        "border_papers": border_papers,
    })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT, "w") as f:
    json.dump(voids_output, f, indent=2)

print(f"\n{'='*60}")
print(f"Saved → {OUTPUT}  ({len(voids_output)} voids)")
print(f"\nTop 5 voids (largest empty circle first):")
for v in voids_output[:5]:
    verts = len(v["shape"]["vertices"])
    print(f'\n  [{v["void_id"]:>2}] "{v["name"]}"')
    print(f"       centroid=({v['centroid'][0]:.2f}, {v['centroid'][1]:.2f})  "
          f"empty_radius={v['empty_radius']:.4f}  "
          f"area={v['shape_area']:.4f}  "
          f"hull_verts={verts}")
    for p in v["border_papers"][:3]:
        print(f"         → {p['title'][:72]}")

print("\nDone.")