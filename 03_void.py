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
# from openrouter import OpenRouter
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

# Parallelism
N_JOBS = -1

# LLM naming — disabled for now
DO_LLM_NAMING = False
LLM_DELAY     = 0.5

# ---------------------------------------------------------------------------
# HackClub client — commented out while LLM naming is disabled
# ---------------------------------------------------------------------------
# HACKCLUB_KEY = os.getenv("HACKCLUB_KEY")
# if DO_LLM_NAMING and not HACKCLUB_KEY:
#     raise EnvironmentError("HACKCLUB_KEY not set — add it to your .env file")
#
# ai_client = OpenRouter(
#     api_key=HACKCLUB_KEY or "dummy",
#     server_url="https://ai.hackclub.com/proxy/v1",
# )

# ---------------------------------------------------------------------------
# LLM: name a void given its border paper titles
# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT = (
#     "You are a research cartographer mapping the landscape of scientific literature. "
#     "You will receive a list of paper titles that SURROUND an empty region in a 2D "
#     "semantic embedding space. Your task is to name the research topic that is "
#     "conspicuously ABSENT — the intellectual gap this void represents.\n\n"
#     "Guidelines:\n"
#     "- The border papers are the EDGES of the hole, not the hole itself.\n"
#     "- The void is what would CONNECT or BRIDGE these surrounding topics but doesn't exist yet.\n"
#     "- Be specific. 'Interdisciplinary research' is not a useful answer.\n"
#     "- The name should read like a real research field or open problem (3-8 words).\n\n"
#     "Respond with ONLY valid JSON, no markdown fences:\n"
#     '{"name": "<short noun phrase>", "reasoning": "<1-2 sentences>"}'
# )

def name_void(border_titles: list, void_id: int) -> dict:
    """Call the LLM and return {"name": ..., "reasoning": ...}."""
    # --- LLM naming disabled: return placeholder ---
    return {"name": f"Void {void_id}", "reasoning": ""}

    # try:
    #     titles_block = "\n".join(f"- {t}" for t in border_titles)
    #     user_msg = (
    #         f"These papers surround Void #{void_id} in the embedding space:\n\n"
    #         f"{titles_block}\n\n"
    #         f"What research topic is conspicuously absent in the gap between them?"
    #     )
    #     response = ai_client.chat.send(
    #         model="openai/gpt-oss-120b",
    #         messages=[
    #             {"role": "system", "content": _SYSTEM_PROMPT},
    #             {"role": "user",   "content": user_msg},
    #         ],
    #     )
    #     raw = response.choices[0].message.content.strip()
    #     if raw.startswith("```"):
    #         raw = raw.split("```")[1]
    #         if raw.startswith("json"):
    #             raw = raw[4:]
    #     raw = raw.strip()
    #     parsed = json.loads(raw)
    #     return {
    #         "name":      str(parsed.get("name", f"Void {void_id}")),
    #         "reasoning": str(parsed.get("reasoning", "")),
    #     }
    # except Exception as e:
    #     print(f"  Warning: naming failed for void {void_id}: {e}")
    #     return {"name": f"Unnamed Void {void_id}", "reasoning": ""}


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