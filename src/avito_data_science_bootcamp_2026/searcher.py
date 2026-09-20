"""Гибридный поиск, комбинирующий Elasticsearch, Qdrant, и RRF."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

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
    """Считает частые слова, которые есть в max_df (50%) документов, с использование CountVectorizer."""
    from sklearn.feature_extraction.text import CountVectorizer

    sample = texts[:50000] if len(texts) > 50000 else texts
    vec = CountVectorizer(min_df=2, max_df=1.0)
    X = vec.fit_transform(sample)
    n_docs = X.shape[0]  # pyright: ignore[reportOptionalSubscript, reportAttributeAccessIssue]

    doc_freqs = np.diff(X.tocsc().indptr)  # pyright: ignore[reportAttributeAccessIssue]
    stopword_indices = np.where(doc_freqs > max_df * n_docs)[0]
    inv_vocab = {v: k for k, v in vec.vocabulary_.items()}
    frequent_words = {inv_vocab[idx] for idx in stopword_indices}
    return frequent_words


# Режим обработки локации в Qdrant-канале:
# - "must": жёсткий фильтр location_id (как в исходном пайплайне);
# - "no_must": фильтр не применяется никогда;
# - "conditional": фильтр снимается, если в локации запроса мало/нет объявлений
#   (loc_counts[loc] < LOCATION_POOL_THRESHOLD).
LOCATION_MODE = "conditional"
# Порог размера пула локации, ниже которого must-фильтр снимается (для conditional).
LOCATION_POOL_THRESHOLD = 50
# Вес Qdrant-канала при снятом must (ES-канал всегда весит 1.0). Для запросов с
# большим пулом локации must сохраняется и вес Qdrant = 1.0.
LOCATION_QD_WEIGHT = 0.3
# Бонус к RRF-скору объявления, локация которого совпадает с локацией запроса.
LOCATION_BONUS = 0.0
# RRF-константа.
RRF_K = 30


class Searcher:
    """Гибридный поиск, комбинирующий Elasticsearch (BM25) и Qdrant (глубокий поиск) с RRF."""

    def __init__(
        self,
        es_indexer: ESIndexer,
        qdrant_indexer: QdrantIndexer,
        embedder: TextEmbedder,
        item_prices: dict[str, float],
        frequent_words: set[str] | None = None,
        loc_counts: dict[Any, int] | None = None,
        item_locations: dict[str, Any] | None = None,
        location_mode: str = LOCATION_MODE,
        location_pool_threshold: int = LOCATION_POOL_THRESHOLD,
        location_qd_weight: float = LOCATION_QD_WEIGHT,
        location_bonus: float = LOCATION_BONUS,
    ) -> None:
        self.es_indexer = es_indexer
        self.qdrant_indexer = qdrant_indexer
        self.embedder = embedder
        self.item_prices = item_prices
        self.frequent_words = frequent_words or set()
        self.loc_counts = loc_counts or {}
        self.item_locations = item_locations or {}
        self.location_mode = location_mode
        self.location_pool_threshold = location_pool_threshold
        self.location_qd_weight = location_qd_weight
        self.location_bonus = location_bonus

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
        """Строит запрос в Elasticsearch."""
        should_clauses: list[dict[str, Any]] = []

        # 1. Главный мульти-метч
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

        # 2. Terms-запрос на params_keys (распаршенных ключах фильтров) с 4x бустом
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

        # 3. Для каждого ключа, terms-запрос по param_<slug> с 3х бустом
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

        # 4. Слова из поля `Слова в описании`: multi_match description^3, title^2
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

        # 5. Буст по локации
        # Пропущен для запросов с доставкой: локация объявления мало влияет на них
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

        # 6. Фильтр по рейтингу
        filter_clauses: list[dict[str, Any]] = []
        if special_params.min_rating is not None:
            filter_clauses.append(
                {"range": {"rating": {"gte": special_params.min_rating}}}
            )

        # Фильтр по макро категории, если он есть и не 0
        if category_id is not None and str(category_id) not in ("0", ""):
            filter_clauses.append({"term": {"category_id": str(category_id)}})

        query_body: dict[str, Any] = {
            "bool": {
                "should": should_clauses if should_clauses else [{"match_all": {}}],
            }
        }
        if filter_clauses:
            query_body["bool"]["filter"] = filter_clauses

        # Сортировка по цене намеренно не применяется тут: сортировка кандидатов по
        # по цене до _score уронит релевантные средне-ценовые объявления в топе.
        # Сортировка по цене происходит позже на финальных топ-50 (см search_batch).
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
        apply_location_filter: bool = True,
    ) -> Filter | None:
        """Строит Qdrant Filter из параметров запроса: category, rating, и location.

        apply_location_filter=False позволяет снять жёсткий must по локации
        (используется для запросов без/с малым пулом локации), сохраняя фильтры
        по рейтингу и категории.
        """
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

        if (
            apply_location_filter
            and location_id is not None
            and str(location_id) not in ("0", "")
        ):
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
        """Делает батчевый поиск по ES и Qdrant в единых HTTP запросах."""
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

        # 1. Парсим параметры для каждого запроса в батче
        parsed_batch: list[tuple[dict[str, list[str]], SpecialParams]] = []
        for p_text in batch_params_texts:
            raw_text = p_text or ""
            p_dict = parse_params(raw_text, KEY_REGEX)
            reg_params, special = extract_special_params(p_dict)
            parsed_batch.append((reg_params, special))

        # 2. Строим и производим батчевый поиск по Elasticsearch (msearch)
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

        # 3. Строим запрос и выполняем батчевый поиск по Qdrant (query_batch_points)
        # Один глубокий проход для запроса:
        # Категория + рейтинг в must, плюс location_id в must
        # для запросов без доставки с валидной локацией
        qdrant_requests: list[QueryRequest] = []
        apply_loc_filters: list[bool] = []
        for i in range(batch_size):
            reg_params, special = parsed_batch[i]
            valid_loc = (
                not is_delivery[i]
                and locations[i] is not None
                and str(locations[i]) not in ("0", "")
            )
            loc_id = locations[i] if valid_loc else None
            apply_loc = self.should_apply_location_filter(loc_id) if valid_loc else True
            apply_loc_filters.append(apply_loc)
            q_filter = self.build_qdrant_filter(
                reg_params,
                special,
                location_id=loc_id,
                category_id=categories[i],
                apply_location_filter=apply_loc,
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

        # 4. Взвешенный RRF с локационным бонусом и Special Sort
        results: list[list[str]] = []
        for i in range(batch_size):
            _, special = parsed_batch[i]
            es_hits = es_batch_results[i] if i < len(es_batch_results) else []
            qdrant_hits = (
                qdrant_batch_results[i] if i < len(qdrant_batch_results) else []
            )

            qd_weight = 1.0 if apply_loc_filters[i] else self.location_qd_weight
            apply_bonus = (
                not is_delivery[i]
                and locations[i] is not None
                and str(locations[i]) not in ("0", "")
            )
            top_candidates = self.weighted_rrf(
                [es_hits, qdrant_hits],
                weights=[1.0, qd_weight],
                k=RRF_K,
                top=top_k,
                query_location=locations[i] if apply_bonus else None,
            )

            results.append(top_candidates)

        return results

    def should_apply_location_filter(self, location_id: Any) -> bool:
        """Нужно ли применять жёсткий must-фильтр по локации для Qdrant.

        В режиме conditional фильтр снимается, если в локации запроса меньше
        location_pool_threshold объявлений корпуса (в т.ч. 0) — тогда must
        отсекал бы весь dense-канал. В режимах must/no_must поведение
        фиксировано.
        """
        if self.location_mode == "no_must":
            return False
        if self.location_mode == "must":
            return True
        # conditional
        if location_id is None or str(location_id) in ("0", ""):
            return True
        pool = self.loc_counts.get(location_id, 0)
        return pool >= self.location_pool_threshold

    def weighted_rrf(
        self,
        rank_lists: list[list[tuple[str, float]]],
        weights: list[float] | None = None,
        k: int = 30,
        top: int = 50,
        query_location: Any = None,
    ) -> list[str]:
        """Взвешенный RRF с опциональным бонусом за совпадение локации.

        Абсолютные скоры каналов не используются (как и в обычном RRF), поэтому
        локационный бонус добавляется к итоговому RRF-скору.
        """
        if weights is None:
            weights = [1.0] * len(rank_lists)
        scores: dict[str, float] = {}
        for rl, w in zip(rank_lists, weights):
            if w == 0.0:
                continue
            for rank, (item_id, _) in enumerate(rl):
                scores[item_id] = scores.get(item_id, 0.0) + w / (k + rank + 1)

        if query_location is not None and str(query_location) not in ("0", ""):
            q_loc = str(query_location)
            for item_id in scores:
                if str(self.item_locations.get(item_id)) == q_loc:
                    scores[item_id] += self.location_bonus

        sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_items[:top]

    @staticmethod
    def rrf(
        rank_lists: list[list[tuple[str, float]]],
        k: int = 30,
        top: int = 50,
    ) -> list[str]:
        """Reciprocal Rank Fusion на нескольких ранжированнных списках из (item_id, score)."""
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
        """Выполняет единичный поиск по запросу."""
        raw_params_text = search_infm_params_text or ""
        embed_text = f"{search_query} {raw_params_text}".strip()
        query_vector = self.embedder.encode([embed_text], show_progress_bar=False)
        return self.search_batch(
            batch_queries=[search_query],
            batch_params_texts=[search_infm_params_text],
            batch_vectors=query_vector,
            top_k=top_k,
        )[0]
