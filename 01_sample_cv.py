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

    print("Loading parquet dataset...")
    df = pd.read_parquet("./hf_data/")
    print(f"Total rows: {len(df):,}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    collected = []
    skipped_no_meta = 0

    print(f"Filtering for categories: {TARGET_CATEGORIES}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doi = str(row["DOI"]).strip()

        category = cat_map.get(doi)  # None if not in Kaggle snapshot — skip

        if category is None:
            skipped_no_meta += 1
            continue

        if category in TARGET_CATEGORIES:
            collected.append(row)
            if len(collected) >= 50_000:
                break

    total_checked = len(collected) + skipped_no_meta + (
        # rows iterated that weren't in target and weren't skipped
        sum(1 for _ in [])  # placeholder; tqdm handles display
    )

    print(f"\nSkipped (not in Kaggle snapshot): {skipped_no_meta:,}")
    print(f"Collected: {len(collected):,}")

    result_df = pd.DataFrame(collected)
    out_path = "sample_cv.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"Saved {len(result_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()