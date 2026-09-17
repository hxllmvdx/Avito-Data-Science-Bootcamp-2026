"""Hybrid search engine combining Elasticsearch, Qdrant, and RRF."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    QueryRequest,
    Range,
)

from avito_data_science_bootcamp_2026.embeddings import TextEmbedder
from avito_data_science_bootcamp_2026.es_indexer import ESIndexer
from avito_data_science_bootcamp_2026.parser import (
    KEY_REGEX,
    SpecialParams,
    extract_special_params,
    parse_params,
    to_slug,
)
from avito_data_science_bootcamp_2026.qdrant_indexer import QdrantIndexer

logger = logging.getLogger(__name__)


def compute_frequent_words(texts: Sequence[str], max_df: float = 0.5) -> set[str]:
    """Compute words that occur in more than max_df (50%) of documents using CountVectorizer."""
    from sklearn.feature_extraction.text import CountVectorizer

    sample = texts[:50000] if len(texts) > 50000 else texts
    vec = CountVectorizer(min_df=2, max_df=1.0)
    X = vec.fit_transform(sample)
    n_docs = X.shape[0]

    doc_freqs = np.diff(X.tocsc().indptr)
    stopword_indices = np.where(doc_freqs > max_df * n_docs)[0]
    inv_vocab = {v: k for k, v in vec.vocabulary_.items()}
    frequent_words = {inv_vocab[idx] for idx in stopword_indices}
    return frequent_words


class Searcher:
    """Hybrid searcher combining Elasticsearch (BM25) and Qdrant (dense vector) via RRF."""

    def __init__(
        self,
        es_indexer: ESIndexer,
        qdrant_indexer: QdrantIndexer,
        embedder: TextEmbedder,
        item_prices: dict[str, float],
        frequent_words: set[str] | None = None,
    ) -> None:
        self.es_indexer = es_indexer
        self.qdrant_indexer = qdrant_indexer
        self.embedder = embedder
        self.item_prices = item_prices
        self.frequent_words = frequent_words or set()

    def build_es_query(
        self,
        search_query: str,
        q_params: dict[str, list[str]],
        special_params: SpecialParams,
        location_id: Any = None,
        category_id: Any = None,
        is_delivery: bool = False,
        size: int = 800,
    ) -> dict[str, Any]:
        """Build Elasticsearch query."""
        should_clauses: list[dict[str, Any]] = []

        # 1. Main multi_match
        clean_search_query = search_query.strip()
        if clean_search_query:
            should_clauses.append(
                {
                    "multi_match": {
                        "query": clean_search_query,
                        "fields": ["title^3", "params_flat^1", "description^2"],
                    }
                }
            )

        # 2. Terms on params_keys with boost 4
        param_keys = list(q_params.keys())
        if param_keys:
            should_clauses.append(
                {
                    "terms": {
                        "params_keys": param_keys,
                        "boost": 4.0,
                    }
                }
            )

        # 3. For each key, terms by param_<slug> with boost 3
        for k, vals in q_params.items():
            if vals:
                slug_field = f"param_{to_slug(k)}"
                should_clauses.append(
                    {
                        "terms": {
                            slug_field: vals,
                            "boost": 3.0,
                        }
                    }
                )

        # 4. Words in description: multi_match description^3, title^2
        if special_params.description_words:
            valid_words = [
                w
                for w in special_params.description_words
                if w.lower() not in self.frequent_words
            ]
            if valid_words:
                words_query = " ".join(valid_words)
                should_clauses.append(
                    {
                        "multi_match": {
                            "query": words_query,
                            "fields": ["description^3", "title^2"],
                        }
                    }
                )

        # 5. Location boost (crucial for classifieds: 84% match).
        # Skipped for delivery searches: item location is weakly tied to them.
        if (
            not is_delivery
            and location_id is not None
            and str(location_id) not in ("0", "")
        ):
            should_clauses.append(
                {
                    "term": {
                        "location_id": {
                            "value": str(location_id),
                            "boost": 8.0,
                        }
                    }
                }
            )

        # 6. Filters
        filter_clauses: list[dict[str, Any]] = []
        if special_params.min_rating is not None:
            filter_clauses.append(
                {"range": {"rating": {"gte": special_params.min_rating}}}
            )

        # Category filter if provided and not 0
        if category_id is not None and str(category_id) not in ("0", ""):
            filter_clauses.append({"term": {"category_id": str(category_id)}})

        query_body: dict[str, Any] = {
            "bool": {
                "should": should_clauses if should_clauses else [{"match_all": {}}],
            }
        }
        if filter_clauses:
            query_body["bool"]["filter"] = filter_clauses

        # Price sort is intentionally NOT applied here: sorting 400 candidates by
        # price before _score would drop relevant mid-price items from the pool.
        # Price re-sorting happens later inside the final top-50 (see search_batch).
        es_sort: list[Any] = ["_score"]

        return {
            "size": size,
            "query": query_body,
            "sort": es_sort,
            "_source": ["item_id"],
        }

    def build_qdrant_filter(
        self,
        q_params: dict[str, list[str]],
        special_params: SpecialParams,
        location_id: Any = None,
        category_id: Any = None,
    ) -> Filter | None:
        """Build Qdrant Filter from query params, category, rating, and location."""
        must_conditions: list[Any] = []

        if special_params.min_rating is not None:
            must_conditions.append(
                FieldCondition(
                    key="rating",
                    range=Range(gte=special_params.min_rating),
                )
            )

        if category_id is not None and str(category_id) not in ("0", ""):
            must_conditions.append(
                FieldCondition(
                    key="category_id",
                    match=MatchValue(value=str(category_id)),
                )
            )

        if location_id is not None and str(location_id) not in ("0", ""):
            must_conditions.append(
                FieldCondition(
                    key="location_id",
                    match=MatchValue(value=str(location_id)),
                )
            )

        if must_conditions:
            return Filter(must=must_conditions)
        return None

    def search_batch(
        self,
        batch_queries: list[str],
        batch_params_texts: list[str | None],
        batch_vectors: np.ndarray,
        batch_locations: list[Any] | None = None,
        batch_categories: list[Any] | None = None,
        batch_is_delivery: list[Any] | None = None,
        top_k: int = 50,
        candidate_size: int = 800,
    ) -> list[list[str]]:
        """Perform batched search across ES and Qdrant in single HTTP calls."""
        batch_size = len(batch_queries)
        if batch_size == 0:
            return []

        locations = (
            batch_locations if batch_locations is not None else [None] * batch_size
        )
        categories = (
            batch_categories if batch_categories is not None else [None] * batch_size
        )
        is_delivery = [
            bool(x) if x is not None else False
            for x in (
                batch_is_delivery
                if batch_is_delivery is not None
                else [None] * batch_size
            )
        ]

        # 1. Parse params for each query in batch
        parsed_batch: list[tuple[dict[str, list[str]], SpecialParams]] = []
        for p_text in batch_params_texts:
            raw_text = p_text or ""
            p_dict = parse_params(raw_text, KEY_REGEX)
            reg_params, special = extract_special_params(p_dict)
            parsed_batch.append((reg_params, special))

        # 2. Build and execute batch Elasticsearch search (msearch)
        es_searches: list[dict[str, Any]] = []
        for i in range(batch_size):
            q_text = batch_queries[i]
            reg_params, special = parsed_batch[i]
            q_body = self.build_es_query(
                q_text,
                reg_params,
                special,
                location_id=locations[i],
                category_id=categories[i],
                is_delivery=is_delivery[i],
                size=candidate_size,
            )
            es_searches.append({"index": self.es_indexer.index_name})
            es_searches.append(q_body)

        es_batch_results: list[list[tuple[str, float]]] = []
        try:
            msearch_resp = self.es_indexer.client.msearch(searches=es_searches)
            for resp in msearch_resp.get("responses", []):
                hits = resp.get("hits", {}).get("hits", [])
                es_batch_results.append(
                    [(h["_id"], float(h.get("_score") or 0.0)) for h in hits]
                )
        except Exception as e:
            logger.error(f"ES msearch failed: {e}")
            es_batch_results = [[] for _ in range(batch_size)]

        # 3. Build and execute batch Qdrant search (query_batch_points)
        # Single dense pass per query:
        # Category + rating in must, plus location_id in must for non-delivery
        # queries with valid location.
        qdrant_requests: list[QueryRequest] = []
        for i in range(batch_size):
            reg_params, special = parsed_batch[i]
            loc_id = (
                locations[i]
                if (
                    not is_delivery[i]
                    and locations[i] is not None
                    and str(locations[i]) not in ("0", "")
                )
                else None
            )
            q_filter = self.build_qdrant_filter(
                reg_params,
                special,
                location_id=loc_id,
                category_id=categories[i],
            )
            qdrant_requests.append(
                QueryRequest(
                    query=batch_vectors[i].tolist(),
                    filter=q_filter,
                    limit=candidate_size,
                    with_payload=["item_id"],
                )
            )

        qdrant_batch_results: list[list[tuple[str, float]]] = []
        try:
            q_batch_resp = self.qdrant_indexer.client.query_batch_points(
                collection_name=self.qdrant_indexer.collection_name,
                requests=qdrant_requests,
            )
            for resp in q_batch_resp:
                qdrant_batch_results.append(
                    [
                        (
                            p.payload["item_id"],
                            float(p.score if p.score is not None else 0.0),
                        )
                        for p in resp.points
                        if p.payload and "item_id" in p.payload
                    ]
                )
        except Exception as e:
            logger.error(f"Qdrant query_batch_points failed: {e}")
            qdrant_batch_results = [[] for _ in range(batch_size)]

        # 4. Symmetric RRF and Special Sort for each query
        results: list[list[str]] = []
        for i in range(batch_size):
            _, special = parsed_batch[i]
            es_hits = es_batch_results[i] if i < len(es_batch_results) else []
            qdrant_hits = (
                qdrant_batch_results[i] if i < len(qdrant_batch_results) else []
            )

            top_candidates = self.rrf([es_hits, qdrant_hits], k=70, top=top_k)

            if special.sort_order and top_candidates:
                is_desc = special.sort_order == "desc"
                top_candidates = sorted(
                    top_candidates,
                    key=lambda x: self.item_prices.get(x, 0.0),
                    reverse=is_desc,
                )

            results.append(top_candidates)

        return results

    @staticmethod
    def rrf(
        rank_lists: list[list[tuple[str, float]]],
        k: int = 30,
        top: int = 50,
    ) -> list[str]:
        """Reciprocal Rank Fusion on multiple ranked lists of (item_id, score)."""
        scores: dict[str, float] = {}
        for rl in rank_lists:
            for rank, (item_id, _) in enumerate(rl):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
        sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_items[:top]

    def search_query(
        self,
        search_query: str,
        search_infm_params_text: str | None,
        top_k: int = 50,
    ) -> list[str]:
        """Execute single search query."""
        raw_params_text = search_infm_params_text or ""
        embed_text = f"{search_query} {raw_params_text}".strip()
        query_vector = self.embedder.encode([embed_text], show_progress_bar=False)
        return self.search_batch(
            batch_queries=[search_query],
            batch_params_texts=[search_infm_params_text],
            batch_vectors=query_vector,
            top_k=top_k,
        )[0]
