import pandas as pd
from tqdm import tqdm

TARGET_CATEGORIES = {"cs.CV", "cs.AI"}

# Path to the Kaggle ArXiv metadata snapshot
# Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv
KAGGLE_SNAPSHOT = "./arxiv-metadata-oai-snapshot.json"


def load_kaggle_category_map(snapshot_path: str) -> dict[str, str]:
    """
    Stream the Kaggle snapshot JSON (one record per line) and build a
    doi -> primary_category dict. Primary category is the first token
    in the space-separated 'categories' field.
    """
    print(f"Loading Kaggle snapshot: {snapshot_path}")
    cat_map = {}
    with tqdm(desc="Reading snapshot") as pbar:
        for chunk in pd.read_json(snapshot_path, lines=True, chunksize=50_000):
            for _, row in chunk.iterrows():
                doi = str(row["id"]).strip()
                cats = str(row.get("categories", "")).strip()
                primary = cats.split()[0] if cats else None
                cat_map[doi] = primary
            pbar.update(len(chunk))
    print(f"Snapshot loaded: {len(cat_map):,} papers indexed")
    return cat_map


def main():
    cat_map = load_kaggle_category_map(KAGGLE_SNAPSHOT)

    # --- Pass 1: load DOI column only to find matching indices ---
    print("\nPass 1: loading DOI column only...")
    dois = pd.read_parquet("./hf_data/", columns=["DOI"])
    print(f"Total rows: {len(dois):,}")

    # Shuffle with fixed seed
    dois = dois.sample(frac=1, random_state=42)

    matched_indices = []
    skipped_no_meta = 0

    print(f"Filtering for categories: {TARGET_CATEGORIES}")
    for idx, row in tqdm(dois.iterrows(), total=len(dois)):
        doi = str(row["DOI"]).strip()
        category = cat_map.get(doi)

        if category is None:
            skipped_no_meta += 1
            continue

        if category in TARGET_CATEGORIES:
            matched_indices.append(idx)
            if len(matched_indices) >= 100_000:
                break

    print(f"\nSkipped (not in Kaggle snapshot): {skipped_no_meta:,}")
    print(f"Matched: {len(matched_indices):,}")

    # Free the category map and DOI-only df before loading full data
    del cat_map, dois

    # --- Pass 2: load full rows only for matched indices ---
    print("\nPass 2: loading full rows for matched papers...")
    df_full = pd.read_parquet("./hf_data/")
    result_df = df_full.loc[matched_indices]
    del df_full

    out_path = "sample_cv.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"Saved {len(result_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()