import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
from tqdm import tqdm

TARGET_CATEGORIES = {"cs.CV", "cs.AI"}
ARXIV_NS = "http://arxiv.org/schemas/atom"
ARXIV_META_NS = "http://arxiv.org/schemas/atom"


def get_primary_category(doi: str) -> str | None:
    """
    Query the ArXiv API for a given DOI/ID and return the primary category term.
    Returns None if the request fails or the category cannot be parsed.
    """
    url = f"http://export.arxiv.org/api/query?id_list={doi}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    try:
        root = ET.fromstring(resp.text)
        # Namespace map used by ArXiv Atom feed
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        cat_el = entry.find("arxiv:primary_category", ns)
        if cat_el is None:
            # Fall back to the standard Atom category element
            cat_el = entry.find("atom:category", ns)
        if cat_el is not None:
            return cat_el.get("term")
    except ET.ParseError:
        return None

    return None


def is_target_category(doi: str) -> bool:
    category = get_primary_category(doi)
    print(f"DOI: {doi}, Category: {category}")
    return category in TARGET_CATEGORIES


def main():
    print("Loading parquet dataset...")
    df = pd.read_parquet("./hf_data/")
    print(f"Total rows: {len(df):,}")

    # Shuffle with a fixed seed so results are reproducible
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    collected = []
    checked = 0

    print(f"Filtering for categories: {TARGET_CATEGORIES}")
    print("Querying ArXiv API (rate-limited to ~3 req/s)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doi = str(row["DOI"]).strip()
        if is_target_category(doi):
            collected.append(row)
            if len(collected) >= 50_000:
                break
        checked += 1
        # ArXiv asks for no more than 3 requests/second
        time.sleep(0.34)

    print(f"\nChecked {checked:,} papers, kept {len(collected):,}")

    result_df = pd.DataFrame(collected)
    out_path = "sample_cv.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"Saved {len(result_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()