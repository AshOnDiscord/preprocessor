"""
03_voids.py
-----------
Detect low-density void regions in 2D UMAP space, find the border papers
that ring each void, compute the void's convex-hull shape, and name each
void using an LLM — "what topic is conspicuously absent here?"

Pipeline:
  1. Fit KDE on 2D UMAP coords
  2. Build a dense candidate grid clipped to the alpha-shape (concave hull)
  3. Filter candidates: inside hull, not too far from real data,
     surrounded in all directions (angle spread check)
  4. Pick top N lowest-density survivors → void centres
  5. Per void: find K nearest real papers → border papers
  6. Per void: convex hull of border paper coords → shape + area
  7. Per void: send border titles to LLM → topic name

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
    "log_density":    -6.123,
    "name":           "Quantum-Biological Energy Transduction",
    "name_reasoning": "Border papers sit at the edge of quantum physics and
                        cell metabolism but nothing bridges the two...",
    "shape": {
      "type":     "convex_hull",
      "vertices": [[x0,y0], [x1,y1], ...]   // ordered CCW, open polygon
    },
    "shape_area":    1.234,                  // UMAP units²
    "border_papers": [
      {"title": "...", "DOI": "...", "x": ..., "y": ..., "cluster": ...},
      ...
    ]
  }

Dependencies:
  pip install pandas numpy scikit-learn alphashape shapely joblib pyarrow python-dotenv
  (openrouter is local — same as used in 02_umap.py)
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
from openrouter import OpenRouter
from shapely.geometry import MultiPoint, Point
from sklearn.neighbors import KernelDensity, NearestNeighbors
import alphashape

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT  = "umap_200k.parquet"
OUTPUT = "voids.json"

N_VOIDS = 50

# KDE
KDE_BANDWIDTH = 0.1

# Alpha shape (concave hull of entire corpus)
ALPHA = 0.05

# Candidate grid
GRID_RES = 300

# Surroundedness filter
N_NEIGHBORS      = 20
MAX_DIST_PCTILE  = 75           # percentile cap on mean neighbour distance
MIN_ANGLE_SPREAD = np.pi * 1.0  # radians — half the circle must be covered
MAX_CV           = 0.5          # max coefficient of variation of neighbour dists

# Border papers per void
BORDER_K = 10

# Parallelism
N_JOBS     = -1
KDE_CHUNKS = max(8, multiprocessing.cpu_count())

# LLM naming
DO_LLM_NAMING = True   # set False to skip (faster, fully offline)
LLM_DELAY     = .5    # seconds between calls to avoid rate limiting

# ---------------------------------------------------------------------------
# HackClub client (matches 02_umap.py exactly)
# ---------------------------------------------------------------------------
HACKCLUB_KEY = os.getenv("HACKCLUB_KEY")
if DO_LLM_NAMING and not HACKCLUB_KEY:
    raise EnvironmentError("HACKCLUB_KEY not set — add it to your .env file")

ai_client = OpenRouter(
    api_key=HACKCLUB_KEY or "dummy",
    server_url="https://ai.hackclub.com/proxy/v1",
)

# ---------------------------------------------------------------------------
# LLM: name a void given its border paper titles
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a research cartographer mapping the landscape of scientific literature. "
    "You will receive a list of paper titles that SURROUND an empty region in a 2D "
    "semantic embedding space. Your task is to name the research topic that is "
    "conspicuously ABSENT — the intellectual gap this void represents.\n\n"
    "Guidelines:\n"
    "- The border papers are the EDGES of the hole, not the hole itself.\n"
    "- The void is what would CONNECT or BRIDGE these surrounding topics but doesn't exist yet.\n"
    "- Be specific. 'Interdisciplinary research' is not a useful answer.\n"
    "- The name should read like a real research field or open problem (3-8 words).\n\n"
    "Respond with ONLY valid JSON, no markdown fences:\n"
    '{"name": "<short noun phrase>", "reasoning": "<1-2 sentences>"}'
)

def name_void(border_titles: list, void_id: int) -> dict:
    """Call the LLM and return {"name": ..., "reasoning": ...}."""
    titles_block = "\n".join(f"- {t}" for t in border_titles)
    user_msg = (
        f"These papers surround Void #{void_id} in the embedding space:\n\n"
        f"{titles_block}\n\n"
        f"What research topic is conspicuously absent in the gap between them?"
    )
    try:
        response = ai_client.chat.send(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return {
            "name":      str(parsed.get("name", f"Void {void_id}")),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as e:
        print(f"  Warning: naming failed for void {void_id}: {e}")
        return {"name": f"Unnamed Void {void_id}", "reasoning": ""}


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
    hull = mp.convex_hull  # Polygon, LineString, or Point if degenerate

    if hull.geom_type == "Polygon":
        coords = list(hull.exterior.coords)[:-1]  # drop repeated closing vertex
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
# Load
# ---------------------------------------------------------------------------
print(f"Reading {INPUT} ...")
df = pd.read_parquet(INPUT)
print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

xy = df[["x", "y"]].values.astype(np.float64)
print(f"  x ∈ [{xy[:,0].min():.2f}, {xy[:,0].max():.2f}]")
print(f"  y ∈ [{xy[:,1].min():.2f}, {xy[:,1].max():.2f}]")

# ---------------------------------------------------------------------------
# Step 1: Fit KDE on real paper locations
# ---------------------------------------------------------------------------
print(f"\nFitting KDE (bandwidth={KDE_BANDWIDTH}) on {len(xy):,} points ...")
t0  = time.time()
kde = KernelDensity(bandwidth=KDE_BANDWIDTH, kernel="gaussian")
kde.fit(xy)
print(f"  Done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Step 2: Build candidate grid clipped to alpha-shape hull
# ---------------------------------------------------------------------------
print(f"\nBuilding {GRID_RES}×{GRID_RES} candidate grid ...")
xx, yy = np.meshgrid(
    np.linspace(xy[:, 0].min(), xy[:, 0].max(), GRID_RES),
    np.linspace(xy[:, 1].min(), xy[:, 1].max(), GRID_RES),
)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

print(f"Computing alpha shape (alpha={ALPHA}) ...")
t0   = time.time()
hull = alphashape.alphashape(xy.tolist(), ALPHA)
print(f"  Hull type : {hull.geom_type}  ({time.time()-t0:.1f}s)")

print(f"Clipping {len(grid_points):,} grid points to hull (n_jobs={N_JOBS}) ...")
t0 = time.time()

def _inside(p):
    return hull.contains(Point(p))

inside_mask = np.array(
    Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_inside)(p) for p in grid_points
    ),
    dtype=bool,
)
candidates = grid_points[inside_mask]
print(f"  {len(candidates):,} candidates inside hull  ({time.time()-t0:.1f}s)")

if len(candidates) == 0:
    raise RuntimeError(
        "No candidates survived the hull clip. "
        "Try lowering ALPHA (e.g. 0.02) or increasing GRID_RES."
    )

# ---------------------------------------------------------------------------
# Step 3: Score and filter candidates
# ---------------------------------------------------------------------------
print(f"\nScoring {len(candidates):,} candidates ...")

# 3a. KDE log-density — chunked parallel
print(f"  KDE scoring in {KDE_CHUNKS} chunks (n_jobs={N_JOBS}) ...")
t0 = time.time()

def _score(chunk):
    return kde.score_samples(chunk)

log_densities = np.concatenate(
    Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_score)(c) for c in np.array_split(candidates, KDE_CHUNKS)
    )
)
print(f"  KDE done in {time.time()-t0:.1f}s")

# 3b. Nearest-neighbour stats
print(f"  NearestNeighbors (k={N_NEIGHBORS}) ...")
t0 = time.time()
nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm="auto", n_jobs=N_JOBS).fit(xy)
distances, indices = nbrs.kneighbors(candidates)
print(f"  NN done in {time.time()-t0:.1f}s")

mean_dists = distances.mean(axis=1)
cv_dists   = distances.std(axis=1) / (mean_dists + 1e-9)

# 3c. Angular spread — fully vectorised
vectors      = xy[indices[:, 1:]] - candidates[:, None, :]
angles       = np.arctan2(vectors[..., 1], vectors[..., 0])
angle_spread = np.ptp(angles, axis=1)

# 3d. Apply filters
dist_threshold = np.percentile(mean_dists, MAX_DIST_PCTILE)
valid_mask = (
    (mean_dists  <  dist_threshold)   &
    (angle_spread > MIN_ANGLE_SPREAD) &
    (cv_dists    <  MAX_CV)
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
# Step 4: Pick top N void centres (lowest density)
# ---------------------------------------------------------------------------
n_to_find      = min(N_VOIDS, len(candidates))
lowest_idxs    = np.argsort(log_densities)[:n_to_find]
void_centres   = candidates[lowest_idxs]
void_densities = log_densities[lowest_idxs]

print(f"\nSelected {len(void_centres)} void centres")
print(f"  log-density ∈ [{void_densities.min():.3f}, {void_densities.max():.3f}]")

# ---------------------------------------------------------------------------
# Step 5: Border papers — single batched kneighbors call
# ---------------------------------------------------------------------------
print(f"\nFinding {BORDER_K} border papers per void ...")
t0 = time.time()
paper_nbrs = NearestNeighbors(n_neighbors=BORDER_K, algorithm="auto", n_jobs=N_JOBS).fit(xy)
_, border_indices = paper_nbrs.kneighbors(void_centres)
print(f"  Done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Steps 6 & 7: Shape + LLM naming, assemble output
# ---------------------------------------------------------------------------
print(f"\nComputing shapes + "
      f"{'naming voids via LLM' if DO_LLM_NAMING else 'skipping LLM (DO_LLM_NAMING=False)'} ...")
if DO_LLM_NAMING:
    print(f"  ({LLM_DELAY}s delay between calls — "
          f"~{LLM_DELAY * len(void_centres) / 60:.1f} min total)")

voids_output = []

for rank, (centre, log_dens, b_idxs) in enumerate(
    zip(void_centres, void_densities, border_indices)
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

    # --- LLM naming ---
    if DO_LLM_NAMING:
        print(f"  [{rank+1:>3}/{len(void_centres)}] naming ...", end=" ", flush=True)
        naming = name_void(border_titles, rank)
        print(f'"{naming["name"]}"')
        if rank < len(void_centres) - 1:
            time.sleep(LLM_DELAY)
    else:
        naming = {"name": f"Void {rank}", "reasoning": ""}

    voids_output.append({
        "void_id":        rank,
        "void_rank":      rank,
        "centroid":       [float(centre[0]), float(centre[1])],
        "log_density":    float(log_dens),
        "name":           naming["name"],
        "name_reasoning": naming["reasoning"],
        "shape": {
            "type":     shape_dict["type"],
            "vertices": shape_dict["vertices"],  # [[x,y], ...] CCW, open polygon
        },
        "shape_area":    shape_area,             # UMAP units²
        "border_papers": border_papers,
    })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(OUTPUT, "w") as f:
    json.dump(voids_output, f, indent=2)

print(f"\n{'='*60}")
print(f"Saved → {OUTPUT}  ({len(voids_output)} voids)")
print(f"\nTop 5 voids (emptiest first):")
for v in voids_output[:5]:
    verts = len(v["shape"]["vertices"])
    print(f'\n  [{v["void_id"]:>2}] "{v["name"]}"')
    print(f"       centroid=({v['centroid'][0]:.2f}, {v['centroid'][1]:.2f})  "
          f"log_dens={v['log_density']:.3f}  "
          f"area={v['shape_area']:.4f}  "
          f"hull_verts={verts}")
    if v["name_reasoning"]:
        print(f"       {v['name_reasoning'][:120]}")
    for p in v["border_papers"][:3]:
        print(f"         → {p['title'][:72]}")

print("\nDone.")