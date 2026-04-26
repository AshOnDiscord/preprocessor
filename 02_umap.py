"""
02_umap.py
----------
Run UMAP on the 200k sample produced by 01_sample.py.
Input  : sample_200k.parquet   (must have "vector", "title" columns)
Output : umap_200k.parquet
  columns: id, title, DOI, x (float32), y (float32), cluster (int16)
Output : cluster_labels.json
  { "0": "Transformer attention mechanisms", "1": "...", ... }

Clusters are computed on raw high-dimensional vectors (before UMAP),
so they reflect semantic similarity rather than 2D geometry.
Cluster labels are generated via hackclub AI (gpt-oss-120b).

Usage:
  pip install umap-learn pyarrow pandas numpy tqdm scikit-learn openrouter python-dotenv
  echo "HACKCLUB_KEY=your_key_here" > .env
  python 02_umap.py
"""

import time
import json
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.cluster import MiniBatchKMeans
from openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()

INPUT         = "sample_200k.parquet"
OUTPUT        = "umap_200k.parquet"
LABELS_OUTPUT = "cluster_labels.json"
USE_GPU       = False
LLM_DELAY     = 2.0  # seconds between LLM calls to avoid rate limiting

HACKCLUB_KEY = os.getenv("HACKCLUB_KEY")
if not HACKCLUB_KEY:
    raise EnvironmentError("HACKCLUB_KEY not set — add it to your .env file")

# ---------------------------------------------------------------------------
# UMAP hyperparameters
# ---------------------------------------------------------------------------
UMAP_PARAMS = dict(
    n_components = 2,
    n_neighbors  = 15,
    min_dist     = 0.1,
    metric       = "cosine",
    n_epochs     = 200,
    random_state = 42,
    verbose      = True,
)

# ---------------------------------------------------------------------------
# K-means hyperparameters
# ---------------------------------------------------------------------------
KMEANS_PARAMS = dict(
    n_clusters   = 50,
    random_state = 42,
    n_init       = 10,
    batch_size   = 4096,
    verbose      = 1,
)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Reading {INPUT} ...")
df = pd.read_parquet(INPUT)
print(f"  {len(df):,} rows, columns: {list(df.columns)}")

print("Extracting vectors ...")
vectors = np.stack(df["vector"].values).astype(np.float32)
print(f"  vectors shape: {vectors.shape}")

# ---------------------------------------------------------------------------
# K-means clustering on raw high-dimensional vectors
# ---------------------------------------------------------------------------
print(f"\nClustering {len(vectors):,} vectors into {KMEANS_PARAMS['n_clusters']} clusters ...")
t0 = time.time()
kmeans = MiniBatchKMeans(**KMEANS_PARAMS)
cluster_labels = kmeans.fit_predict(vectors).astype(np.int16)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"Cluster sizes — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.0f}")

# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------
if USE_GPU:
    try:
        from cuml.manifold import UMAP
        print("Using cuML UMAP (GPU)")
    except ImportError:
        print("cuML not found — falling back to umap-learn (CPU)")
        from umap import UMAP
        USE_GPU = False
else:
    from umap import UMAP
    print("Using umap-learn (CPU)")

reducer = UMAP(**UMAP_PARAMS)
print(f"\nFitting UMAP on {len(vectors):,} × {vectors.shape[1]} matrix ...")
t0 = time.time()
xy = reducer.fit_transform(vectors)
elapsed = time.time() - t0
print(f"Done in {elapsed/60:.1f} min  ({elapsed:.0f}s)")

x_coords = xy[:, 0].astype(np.float32)
y_coords = xy[:, 1].astype(np.float32)

print(f"\nProjected range — x: [{x_coords.min():.2f}, {x_coords.max():.2f}]  "
      f"y: [{y_coords.min():.2f}, {y_coords.max():.2f}]")

# ---------------------------------------------------------------------------
# Generate cluster labels via hackclub AI
# ---------------------------------------------------------------------------
print("\nGenerating cluster labels via gpt-oss-120b ...")
print(f"({LLM_DELAY}s delay between calls — ~{LLM_DELAY * KMEANS_PARAMS['n_clusters'] / 60:.1f} min total)")

ai_client = OpenRouter(
    api_key=HACKCLUB_KEY,
    server_url="https://ai.hackclub.com/proxy/v1",
)

# Group titles by cluster
df_temp = df[["title"]].copy()
df_temp["cluster"] = cluster_labels
cluster_title_groups = df_temp.groupby("cluster")["title"].apply(list).to_dict()

cluster_name_map = {}
n_clusters = len(cluster_title_groups)

for i, (cid, titles) in enumerate(sorted(cluster_title_groups.items())):
    # Sample up to 30 titles, preferring longer/more descriptive ones
    sample_titles = sorted(titles, key=len, reverse=True)[:30]
    titles_str = "\n".join(f"- {t}" for t in sample_titles)

    prompt = (
        f"Below are titles of research papers grouped by semantic similarity. "
        f"Give a concise 4-6 word label capturing their common theme. "
        f"Reply with the label only, no punctuation, no explanation.\n\n"
        f"{titles_str}\n\nLabel:"
    )

    try:
        response = ai_client.chat.send(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        label = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Warning: labeling failed for cluster {cid}: {e}")
        label = f"Cluster {cid}"

    cluster_name_map[str(cid)] = label
    print(f"  [{i + 1}/{n_clusters}] Cluster {cid}: {label}")

    # Delay between calls to avoid rate limiting (skip after last call)
    if i < n_clusters - 1:
        time.sleep(LLM_DELAY)

# Save labels
with open(LABELS_OUTPUT, "w") as f:
    json.dump(cluster_name_map, f, indent=2)
print(f"\nCluster labels saved → {LABELS_OUTPUT}")

# ---------------------------------------------------------------------------
# Save parquet (drop vectors — visualizer doesn't need them)
# ---------------------------------------------------------------------------
out_df = df[["id", "title", "DOI"]].copy()
out_df["x"]       = x_coords
out_df["y"]       = y_coords
out_df["cluster"] = cluster_labels   # int16, 0-indexed

table = pa.Table.from_pandas(out_df, preserve_index=False)
pq.write_table(table, OUTPUT, compression="snappy")

size_mb = sum(
    pq.ParquetFile(OUTPUT).metadata.row_group(i).total_byte_size
    for i in range(pq.ParquetFile(OUTPUT).metadata.num_row_groups)
) / 1e6

print(f"\nSaved → {OUTPUT}")
print(f"Columns : {list(out_df.columns)}")
print(f"Rows    : {len(out_df):,}")
print(f"Size    : {size_mb:.1f} MB on disk")
print("\nSample rows:")
print(out_df.head(5).to_string(index=False))