"""
03_voids.py
-----------
Detect low-density void regions in 2D UMAP space and find the papers
that form the border around each void.

Detection strategy (matches compute_lowdensity_points in the Flask app):
  1. Fit KDE on 2D UMAP coords
  2. Build a dense candidate grid clipped to the alpha-shape (concave hull)
  3. Filter candidates: must be inside the hull, not too far from real data,
     and reasonably surrounded by real points (angle spread check)
  4. Pick the top N lowest-density survivors — these are the void centres
  5. For each void centre, find the K nearest real papers — these are the
     "border papers" that ring the hole

No DBSCAN grouping: each of the N points is its own discrete void, just
like the Flask endpoint does it. This keeps things simple and predictable.

Input  : umap_200k.parquet   (columns: id, title, DOI, x, y, cluster)
Output : voids.json
  [
    {
      "void_id": 0,
      "void_rank": 0,
      "centroid": [x, y],
      "log_density": -6.123,
      "border_papers": [
        {"title": "...", "DOI": "...", "x": ..., "y": ..., "cluster": ...},
        ...
      ]
    },
    ...
  ]

Usage:
  pip install pandas numpy scikit-learn alphashape shapely tqdm pyarrow
  python 03_voids.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity, NearestNeighbors
import alphashape
from shapely.geometry import Point
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config  (mirrors compute_lowdensity_points defaults, with small improvements)
# ---------------------------------------------------------------------------
INPUT         = "umap_200k.parquet"
OUTPUT        = "voids.json"

# How many void centres to find
N_VOIDS       = 50          # same default as the Flask endpoint

# KDE
KDE_BANDWIDTH = 0.1         # same as Flask — sharp, doesn't blur distinct voids

# Alpha shape (concave hull around real data)
ALPHA         = 0.05        # same as Flask — lower = tighter hull

# Candidate grid
GRID_RES      = 300         # grid points per axis (300×300 = 90k candidates)
                             # increase to 500 for finer void localisation

# Surroundedness filter — keeps only candidates that have real papers on
# most sides, so we don't pick up edge/exterior points by mistake
N_NEIGHBORS        = 20     # same as Flask
MAX_DIST_PCTILE    = 75     # drop candidates farther than this percentile
MIN_ANGLE_SPREAD   = np.pi * 1.0   # ~180° — less strict than Flask's 1.5×π
                                    # (Flask's π×1.5 was killing too many candidates)
MAX_CV             = 0.5    # max coefficient of variation of neighbour distances
                             # (Flask used 0.3 which was very strict; 0.5 is safer)

# Border papers
BORDER_K      = 10          # number of nearest real papers per void

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Reading {INPUT} ...")
df = pd.read_parquet(INPUT)
print(f"  {len(df):,} rows  columns: {list(df.columns)}")

xy = df[["x", "y"]].values.astype(np.float64)
print(f"  x range: [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]")
print(f"  y range: [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")

# ---------------------------------------------------------------------------
# Step 1: Fit KDE on real paper locations
# ---------------------------------------------------------------------------
print(f"\nFitting KDE (bandwidth={KDE_BANDWIDTH}) on {len(xy):,} points ...")
kde = KernelDensity(bandwidth=KDE_BANDWIDTH, kernel="gaussian")
kde.fit(xy)

# ---------------------------------------------------------------------------
# Step 2: Build candidate grid clipped to alpha shape
# ---------------------------------------------------------------------------
print(f"\nBuilding {GRID_RES}×{GRID_RES} candidate grid ...")
xx, yy = np.meshgrid(
    np.linspace(xy[:, 0].min(), xy[:, 0].max(), GRID_RES),
    np.linspace(xy[:, 1].min(), xy[:, 1].max(), GRID_RES),
)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T   # (GRID_RES², 2)

print(f"Computing alpha shape (alpha={ALPHA}) ...")
hull = alphashape.alphashape(xy.tolist(), ALPHA)
print(f"  Hull type: {hull.geom_type}")

print(f"Clipping {len(grid_points):,} grid points to hull ...")
inside_mask = np.array(
    [hull.contains(Point(p)) for p in tqdm(grid_points, desc="  Hull check")],
    dtype=bool,
)
candidates = grid_points[inside_mask]
print(f"  {len(candidates):,} candidates inside hull")

if len(candidates) == 0:
    raise RuntimeError(
        "No candidates survived the hull clip. Try lowering ALPHA (e.g. 0.02) "
        "or increasing GRID_RES."
    )

# ---------------------------------------------------------------------------
# Step 3: Score and filter candidates
# ---------------------------------------------------------------------------
print(f"\nScoring {len(candidates):,} candidates ...")

# 3a. KDE log-density at each candidate (lower = sparser = more void-like)
log_densities = kde.score_samples(candidates)

# 3b. Nearest-neighbour stats from real data points
nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto").fit(xy)
distances, indices = nbrs.kneighbors(candidates)

mean_dists = distances.mean(axis=1)
cv_dists   = distances.std(axis=1) / (mean_dists + 1e-9)

# 3c. Angular spread — are real papers spread around most directions?
#     If a candidate is near the edge of the data cloud, neighbours will all
#     be on one side; we want candidates surrounded on multiple sides.
vectors      = xy[indices[:, 1:]] - candidates[:, None, :]   # (C, N-1, 2)
angles       = np.arctan2(vectors[..., 1], vectors[..., 0])  # (C, N-1)
angle_spread = np.ptp(angles, axis=1)                        # range of angles

# 3d. Apply filters
dist_threshold = np.percentile(mean_dists, MAX_DIST_PCTILE)

valid_mask = (
    (mean_dists  <  dist_threshold)   &   # not too far from real data
    (angle_spread > MIN_ANGLE_SPREAD) &   # surrounded on most sides
    (cv_dists    <  MAX_CV)               # fairly uniform spacing to neighbours
)

candidates    = candidates[valid_mask]
log_densities = log_densities[valid_mask]
print(f"  {valid_mask.sum():,} candidates pass surroundedness filter")

if len(candidates) == 0:
    raise RuntimeError(
        "No candidates survived the surroundedness filter. Try:\n"
        "  - Lowering MIN_ANGLE_SPREAD (e.g. np.pi * 0.8)\n"
        "  - Raising MAX_CV (e.g. 0.7)\n"
        "  - Raising MAX_DIST_PCTILE (e.g. 85)"
    )

# ---------------------------------------------------------------------------
# Step 4: Pick top N lowest-density points — these are the void centres
#          (same logic as the Flask endpoint: just argsort + slice)
# ---------------------------------------------------------------------------
n_to_find = min(N_VOIDS, len(candidates))
lowest_idxs   = np.argsort(log_densities)[:n_to_find]
void_centres  = candidates[lowest_idxs]      # (N_VOIDS, 2)
void_densities = log_densities[lowest_idxs]  # (N_VOIDS,)

print(f"\nSelected {len(void_centres)} void centres")
print(f"  log-density range: [{void_densities.min():.3f}, {void_densities.max():.3f}]")

# ---------------------------------------------------------------------------
# Step 5: For each void centre, find K nearest real papers (border papers)
# ---------------------------------------------------------------------------
print(f"\nFinding {BORDER_K} border papers per void ...")

paper_nbrs = NearestNeighbors(n_neighbors=BORDER_K, algorithm="auto").fit(xy)

voids_output = []

for rank, (centre, log_dens) in enumerate(zip(void_centres, void_densities)):
    dists, idxs = paper_nbrs.kneighbors(centre.reshape(1, -1))
    border_idxs = idxs[0]

    border_papers = []
    for idx in border_idxs:
        row = df.iloc[idx]
        border_papers.append({
            "title":   str(row["title"]),
            "DOI":     str(row["DOI"])     if "DOI"     in df.columns else "",
            "x":       float(row["x"]),
            "y":       float(row["y"]),
            "cluster": int(row["cluster"]) if "cluster" in df.columns else -1,
        })

    voids_output.append({
        "void_id":      rank,          # 0-indexed, ordered by density (emptiest first)
        "void_rank":    rank,
        "centroid":     [float(centre[0]), float(centre[1])],
        "log_density":  float(log_dens),
        "border_papers": border_papers,
    })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT, "w") as f:
    json.dump(voids_output, f, indent=2)

print(f"\n{'='*60}")
print(f"Saved → {OUTPUT}")
print(f"Total voids found : {len(voids_output)}")
print(f"\nTop 5 voids (emptiest first):")
for v in voids_output[:5]:
    print(f"  Void {v['void_id']:>3}  "
          f"centroid=({v['centroid'][0]:.2f}, {v['centroid'][1]:.2f})  "
          f"log_density={v['log_density']:.3f}")
    for p in v["border_papers"][:3]:
        print(f"    → {p['title'][:70]}")
    print()

print("Done.")