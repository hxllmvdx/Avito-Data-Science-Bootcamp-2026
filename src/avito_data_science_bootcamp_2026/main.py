"""Main execution pipeline for hybrid search and evaluation."""

from __future__ import annotations

import csv
import logging
import os
import warnings
from pathlib import Path

# Suppress library warnings and noisy outputs
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers.utils import logging as tf_logging

tf_logging.disable_progress_bar()
tf_logging.set_verbosity_error()

for noisy_logger in [
    "httpx",
    "httpcore",
    "elastic_transport",
    "urllib3",
    "transformers",
    "huggingface_hub",
    "qdrant_client",
    "elasticsearch",
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

import polars as pl
from tqdm import tqdm

from avito_data_science_bootcamp_2026.embeddings import TextEmbedder
from avito_data_science_bootcamp_2026.es_indexer import ESIndexer
from avito_data_science_bootcamp_2026.parser import preprocess_items_df
from avito_data_science_bootcamp_2026.qdrant_indexer import QdrantIndexer
from avito_data_science_bootcamp_2026.searcher import Searcher, compute_frequent_words

logging.basicConfig(level=logging.ERROR)


def evaluate_recall(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """Compute Recall@50 = mean_over_queries( |top50 & relevant| / |relevant| )."""
    recalls: list[float] = []
    for qid, relevant_set in ground_truth.items():
        if not relevant_set:
            continue
        predicted_list = predictions.get(qid, [])
        hits = len(relevant_set.intersection(predicted_list))
        recall = hits / len(relevant_set)
        recalls.append(recall)

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def run_pipeline() -> None:
    """Execute the complete indexing, validation, and benchmarking pipeline."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    items_path = data_dir / "benchmark_items.parquet"
    cached_items_path = data_dir / "items_parsed.parquet"
    queries_path = data_dir / "benchmark_queries.parquet"
    train_path = data_dir / "train.parquet"

    answer_path = output_dir / "answer.csv"
    train_preds_path = output_dir / "train_predictions.csv"

    model_path = base_dir / "finetuned_rubert"

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found at {items_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Benchmark queries file not found at {queries_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Finetuned model dir not fountd at {model_path}")

    # 1. Initialize and verify DB connections
    es_indexer = ESIndexer()
    qdrant_indexer = QdrantIndexer()

    if not es_indexer.check_health():
        raise ConnectionError(
            "Elasticsearch is not reachable at http://localhost:9200. "
            "Please ensure services are running: docker compose up -d"
        )
    if not qdrant_indexer.check_health():
        raise ConnectionError(
            "Qdrant is not reachable at http://localhost:6333. "
            "Please ensure services are running: docker compose up -d"
        )

    # 2. Load and preprocess items
    items_raw_df = pl.read_parquet(items_path)
    items_df = preprocess_items_df(items_raw_df, cache_path=cached_items_path)

    item_prices: dict[str, float] = dict(
        zip(
            items_df["item_id"].to_list(),
            items_df["item_price"].fill_null(0.0).to_list(),
        )
    )

    # 3. Initialize Embedder
    embedder = TextEmbedder(str(model_path.absolute()))

    cached_embeddings_path = data_dir / "item_embeddings.npy"

    # 4. Index documents in ES and Qdrant
    es_indexer._load_up(items_df)
    qdrant_indexer._load_up(
        items_df,
        embedder=embedder,
        embeddings_cache_path=cached_embeddings_path,
    )

    # 5. Frequent words for 'Слова в описании' filter
    frequent_words = compute_frequent_words(
        items_df["item_description_raw"].fill_null("").to_list()
    )

    searcher = Searcher(
        es_indexer=es_indexer,
        qdrant_indexer=qdrant_indexer,
        embedder=embedder,
        item_prices=item_prices,
        frequent_words=frequent_words,
    )

    # 6. Validation on train.parquet (2000 sampled queries)
    if train_path.exists():
        train_df = pl.read_parquet(train_path)

        indexed_item_ids = set(items_df["item_id"].to_list())
        train_filtered = train_df.filter(pl.col("item_id").is_in(indexed_item_ids))

        unique_train_queries = train_filtered.group_by(
            [
                "search_query",
                "search_infm_params_text",
                "search_location_id",
                "search_category",
                "search_is_delivery_search",
            ]
        ).agg(pl.col("item_id").alias("relevant_items"))

        # Sample 2000 queries for fast and representative evaluation
        n_sample = min(2000, len(unique_train_queries))
        eval_train_queries = unique_train_queries.sample(n=n_sample, seed=42)

        train_query_rows = eval_train_queries.to_dicts()
        n_train = len(train_query_rows)

        # Pre-embed clean query texts in batches
        all_train_embed_texts = [
            f"{r['search_query'] or ''} {r['search_infm_params_text'] or ''}".strip()
            for r in train_query_rows
        ]
        train_vectors = embedder.encode(
            all_train_embed_texts, batch_size=512, show_progress_bar=False
        )

        train_predictions: dict[str, list[str]] = {}
        ground_truth: dict[str, set[str]] = {}

        search_chunk_size = 100
        pbar_train = tqdm(total=n_train, desc="Train evaluation")

        for start_idx in range(0, n_train, search_chunk_size):
            end_idx = min(start_idx + search_chunk_size, n_train)
            chunk_rows = train_query_rows[start_idx:end_idx]
            chunk_queries = [r["search_query"] or "" for r in chunk_rows]
            chunk_params = [r["search_infm_params_text"] for r in chunk_rows]
            chunk_locs = [r["search_location_id"] for r in chunk_rows]
            chunk_cats = [r["search_category"] for r in chunk_rows]
            chunk_delivery = [r.get("search_is_delivery_search") for r in chunk_rows]
            chunk_vecs = train_vectors[start_idx:end_idx]

            batch_results = searcher.search_batch(
                batch_queries=chunk_queries,
                batch_params_texts=chunk_params,
                batch_vectors=chunk_vecs,
                batch_locations=chunk_locs,
                batch_categories=chunk_cats,
                batch_is_delivery=chunk_delivery,
                top_k=50,
                candidate_size=400,
            )

            for i, res_items in enumerate(batch_results):
                row_idx = start_idx + i
                qid = f"train_q_{row_idx}"
                train_predictions[qid] = res_items
                ground_truth[qid] = set(chunk_rows[i]["relevant_items"])

            pbar_train.update(len(chunk_rows))

        pbar_train.close()

        train_recall = evaluate_recall(train_predictions, ground_truth)
        tqdm.write(f"TRAIN Recall@50: {train_recall:.4f}")

        # Save train_predictions.csv
        with open(train_preds_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "answer"])
            for qid, preds in train_predictions.items():
                writer.writerow([qid, " ".join(preds[:50])])

    # 7. Inference on benchmark_queries.parquet
    bench_queries_df = pl.read_parquet(queries_path)
    bench_rows = bench_queries_df.to_dicts()
    n_bench = len(bench_rows)

    all_bench_embed_texts = [
        f"{r['search_query'] or ''} {r.get('search_infm_params_text') or ''}".strip()
        for r in bench_rows
    ]
    bench_vectors = embedder.encode(
        all_bench_embed_texts, batch_size=512, show_progress_bar=False
    )

    bench_predictions: list[tuple[str, str]] = []
    pbar_bench = tqdm(total=n_bench, desc="Benchmark search")

    search_chunk_size = 100
    for start_idx in range(0, n_bench, search_chunk_size):
        end_idx = min(start_idx + search_chunk_size, n_bench)
        chunk_rows = bench_rows[start_idx:end_idx]
        chunk_queries = [str(r["search_query"] or "") for r in chunk_rows]
        chunk_params = [r.get("search_infm_params_text") for r in chunk_rows]
        chunk_locs = [r.get("search_location_id") for r in chunk_rows]
        chunk_cats = [r.get("search_category") for r in chunk_rows]
        chunk_delivery = [r.get("search_is_delivery_search") for r in chunk_rows]
        chunk_vecs = bench_vectors[start_idx:end_idx]

        batch_results = searcher.search_batch(
            batch_queries=chunk_queries,
            batch_params_texts=chunk_params,
            batch_vectors=chunk_vecs,
            batch_locations=chunk_locs,
            batch_categories=chunk_cats,
            batch_is_delivery=chunk_delivery,
            top_k=50,
            candidate_size=800,
        )

        for i, res_items in enumerate(batch_results):
            qid = str(chunk_rows[i]["query_id"])
            answer_str = " ".join(res_items[:50])
            bench_predictions.append((qid, answer_str))

        pbar_bench.update(len(chunk_rows))

    pbar_bench.close()

    # Save to output/answer.csv
    with open(answer_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer"])
        for qid, ans in bench_predictions:
            writer.writerow([qid, ans])


def main() -> None:
    """Entrypoint function."""
    run_pipeline()


if __name__ == "__main__":
    main()
