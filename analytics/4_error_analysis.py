"""Error analysis: где теряется Recall@50 на eval и почему разрыв eval->тест ~0.05.

Запуск:
    uv run python analytics/4_error_analysis.py

Скрипт автономный: реконструирует детерминированный eval-срез (те же правила, что
в main.py), прогоняет поиск по уже поднятым ES/Qdrant, классифицирует потери и
печатает все числа из разделов 1-4. Промежуточные артефакты кэшируются в output/.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import polars as pl
from tqdm import tqdm

from avito_data_science_bootcamp_2026.embeddings import TextEmbedder
from avito_data_science_bootcamp_2026.es_indexer import ESIndexer
from avito_data_science_bootcamp_2026.parser import (
    KEY_REGEX,
    extract_special_params,
    parse_params,
)
from avito_data_science_bootcamp_2026.qdrant_indexer import QdrantIndexer
from avito_data_science_bootcamp_2026.searcher import Searcher

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_KEYS = [
    "search_query",
    "search_infm_params_text",
    "search_location_id",
    "search_category",
    "search_is_delivery_search",
]
N_EVAL = 2000
CANDIDATE_SIZE = 800
RRF_K = 30


# --------------------------------------------------------------------------------------
# Утилиты
# --------------------------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def rel_diff(bench: float, train: float) -> str:
    if train == 0:
        return "n/a"
    return f"{(bench - train) / train * 100:+.1f}%"


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_query_embed_cache() -> tuple[dict[str, np.ndarray], object]:
    cache: dict[str, np.ndarray] = {}
    path = DATA_DIR / "query_embed_cache.npz"
    if path.exists():
        loaded = np.load(path, allow_pickle=True)
        cache = {k: loaded[k] for k in loaded.files}
        print(f"[cache] loaded {len(cache)} query embeddings")
    return cache


# --------------------------------------------------------------------------------------
# Раздел 1. Сравнение распределений benchmark_queries vs train
# --------------------------------------------------------------------------------------
def section1(train: pl.DataFrame, bench: pl.DataFrame) -> dict:
    section("РАЗДЕЛ 1. СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ benchmark_queries vs train")

    # train на уровне уникальных запросов + сэмпл равного с benchmark размера
    train_uniq = train.group_by(GROUP_KEYS).agg(pl.len().alias("n"))
    train_q = train_uniq.sample(n=min(len(bench), len(train_uniq)), seed=SEED)
    nb, nt = len(bench), len(train_q)

    def share(df: pl.DataFrame, expr: pl.Expr) -> float:
        return float(df.select(expr.mean()).item())

    rows = []

    # 1.1 delivery
    b = share(bench, (pl.col("search_is_delivery_search") == 1).cast(pl.Float64))
    t = share(train_q, (pl.col("search_is_delivery_search") == 1).cast(pl.Float64))
    rows.append(("1.1 is_delivery_search=True", f"{b:.4f}", f"{t:.4f}", rel_diff(b, t)))

    # 1.2 empty params
    expr_empty = (pl.col("search_infm_params_text").fill_null("") == "").cast(pl.Float64)
    b = share(bench, expr_empty)
    t = share(train_q, expr_empty)
    rows.append(("1.2 empty params text", f"{b:.4f}", f"{t:.4f}", rel_diff(b, t)))

    # 1.3 length
    for label, expr in [
        ("1.3 median chars", pl.col("search_query").fill_null("").str.len_chars().median()),
        ("1.3 p90 chars", pl.col("search_query").fill_null("").str.len_chars().quantile(0.9)),
        ("1.3 median tokens", pl.col("search_query").fill_null("").str.split(" ").list.len().median()),
        ("1.3 p90 tokens", pl.col("search_query").fill_null("").str.split(" ").list.len().quantile(0.9)),
    ]:
        bv = float(bench.select(expr).item())
        tv = float(train_q.select(expr).item())
        rows.append((label, f"{bv:.1f}", f"{tv:.1f}", rel_diff(bv, tv)))

    # 1.5 location concentration
    def loc_top10(df: pl.DataFrame) -> float:
        vc = df["search_location_id"].value_counts(sort=True)
        return float(vc.head(10)["count"].sum()) / len(df)

    b, t = loc_top10(bench), loc_top10(train_q)
    rows.append(("1.5 top-10 location share", f"{b:.4f}", f"{t:.4f}", rel_diff(b, t)))
    rows.append(("1.5 unique locations", f"{bench['search_location_id'].n_unique()}", f"{train_q['search_location_id'].n_unique()}", "n/a"))

    # 1.7 latin / digits
    for label, expr in [
        ("1.7 latin share", pl.col("search_query").fill_null("").str.contains(r"[A-Za-z]").cast(pl.Float64).mean()),
        ("1.7 digit/price share", pl.col("search_query").fill_null("").str.contains(r"\d").cast(pl.Float64).mean()),
        ("1.7 emoji share", pl.col("search_query").fill_null("").str.contains(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]").cast(pl.Float64).mean()),
    ]:
        bv = float(bench.select(expr).item())
        tv = float(train_q.select(expr).item())
        rows.append((label, f"{bv:.4f}", f"{tv:.4f}", rel_diff(bv, tv)))

    print(f"\n{'метрика':<32}{'benchmark':>12}{'train':>12}{'отн.разница':>15}")
    print("-" * 71)
    for label, bv, tv, d in rows:
        print(f"{label:<32}{bv:>12}{tv:>12}{d:>15}")
    print(f"\nn_benchmark={nb}, n_train_sample={nt} (уникальные запросы, seed={SEED})")

    # 1.4 категории
    section("1.4 Категории: топ-15 + хвост")
    for name, df in [("BENCH", bench), ("TRAIN", train_q)]:
        vc = df["search_category"].value_counts(sort=True)
        total = len(df)
        print(f"\n{name}: unique={vc.height}")
        for r in vc.head(15).to_dicts():
            print(f"  cat={r['search_category']:<8} n={r['count']:<6} share={r['count']/total:.4f}")
        tail = vc.slice(15)
        if tail.height:
            print(f"  хвост (cat >15): n={int(tail['count'].sum())} share={float(tail['count'].sum())/total:.4f}")

    # 1.5 пустой пул локации
    items_loc = pl.read_parquet(DATA_DIR / "items_parsed.parquet", columns=["item_location_id"])
    loc_counts = {r["item_location_id"]: r["len"] for r in items_loc.group_by("item_location_id").agg(pl.len().alias("len")).to_dicts()}

    def empty_pool_share(df: pl.DataFrame) -> float:
        sizes = [loc_counts.get(l, 0) for l in df["search_location_id"].to_list()]
        return float(np.mean(np.array(sizes) == 0))

    be, te = empty_pool_share(bench), empty_pool_share(train_q)
    print(f"\n1.5 доля запросов с нулевым пулом локации: bench={be:.4f} train={te:.4f} ({rel_diff(be, te)})")

    # 1.6 точное пересечение текстов
    section("1.6 Пересечение текстов benchmark и train (lower + strip)")
    train_texts = set(train["search_query"].str.to_lowercase().str.strip_chars().to_list())
    bench_norm = bench.with_columns(pl.col("search_query").str.to_lowercase().str.strip_chars().alias("nq"))
    seen = bench_norm.filter(pl.col("nq").is_in(list(train_texts)))
    seen_share = len(seen) / len(bench_norm)
    print(f"benchmark всего: {len(bench_norm)}")
    print(f"текст встречается в train: {len(seen)} ({seen_share:.4f})")

    bi_ids = set(pl.read_parquet(DATA_DIR / "benchmark_items.parquet", columns=["item_id"])["item_id"].to_list())

    # Ключевой факт: какую долю кликов train вообще можно найти в корпусе benchmark.
    train_items = set(train["item_id"].unique().to_list())
    overlap = len(train_items & bi_ids)
    print("\n1.6' Пересечение корпусов:")
    print(f"  уникальных айтемов train: {len(train_items)}")
    print(f"  айтемов benchmark_items:  {len(bi_ids)}")
    print(f"  пересечение:              {overlap} ({overlap/len(train_items):.4f} от train)")
    print(f"  строк train с айтемом в корпусе: {float(train['item_id'].is_in(list(bi_ids)).mean()):.4f}")

    train_clicks = {
        r["nq"]: set(r["ids"])
        for r in train.with_columns(pl.col("search_query").str.to_lowercase().str.strip_chars().alias("nq"))
        .group_by("nq")
        .agg(pl.col("item_id").alias("ids"))
        .to_dicts()
    }
    seen_with_click_in_corpus = 0
    for r in seen.to_dicts():
        clicks = train_clicks.get(r["nq"], set())
        if clicks & bi_ids:
            seen_with_click_in_corpus += 1
    print(f"из seen у которых кликнутый айтем есть в benchmark_items: {seen_with_click_in_corpus} "
          f"({seen_with_click_in_corpus / max(len(seen), 1):.4f})")

    return {"bench": bench, "train_q": train_q, "loc_counts": loc_counts, "bi_ids": bi_ids, "train_clicks": train_clicks}


# --------------------------------------------------------------------------------------
# Раздел 2. Псевдо-оценка на пересечении benchmark ∩ train
# --------------------------------------------------------------------------------------
def section2(train: pl.DataFrame, bench: pl.DataFrame, bi_ids: set, train_clicks: dict) -> None:
    section("РАЗДЕЛ 2. ПСЕВДО-ОЦЕНКА НА ПЕРЕСЕЧЕНИИ benchmark ∩ train")

    ans: dict[str, set[str]] = {}
    with open(OUTPUT_DIR / "answer.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans[row["query_id"]] = set(row["answer"].split())
    print(f"answer.csv: {len(ans)} запросов (предсказания пайплайна на benchmark)")

    bench_norm = bench.with_columns(pl.col("search_query").str.to_lowercase().str.strip_chars().alias("nq"))

    # 2.2 text-only пересечение
    rec_all, n_in_corpus = [], 0
    for r in bench_norm.to_dicts():
        clicks = train_clicks.get(r["nq"])
        if not clicks:
            continue
        pred = ans[r["query_id"]]
        rec_all.append(len(pred & clicks) / len(clicks))
        if clicks & bi_ids:
            n_in_corpus += 1
    print(f"\n2.2 text-only: запросов seen={len(rec_all)}, pseudo-Recall@50 = {np.mean(rec_all):.4f}")

    # 2.2 full-combo пересечение (то же определение запроса, что у eval-среза)
    tq = (
        train.with_columns(pl.col("search_query").str.to_lowercase().str.strip_chars().alias("nq"))
        .group_by(["nq"] + GROUP_KEYS[1:])
        .agg(pl.col("item_id").alias("ids"))
    )
    combo = {
        (r["nq"], r["search_infm_params_text"], r["search_location_id"], r["search_category"], r["search_is_delivery_search"]): set(r["ids"])
        for r in tq.to_dicts()
    }
    rec_combo = []
    for r in bench_norm.to_dicts():
        key = (r["nq"], r["search_infm_params_text"], r["search_location_id"], r["search_category"], r["search_is_delivery_search"])
        clicks = combo.get(key)
        if not clicks:
            continue
        in_corpus = {c for c in clicks if c in bi_ids}
        if not in_corpus:
            continue
        rec_combo.append(len(ans[r["query_id"]] & in_corpus) / len(in_corpus))
    print(f"2.2 full-combo (клики только в корпусе): запросов={len(rec_combo)}, pseudo-Recall@50 = {np.mean(rec_combo):.4f}")

    # 2.4 seen по срезам
    section("2.4 Доля seen по срезам benchmark")
    seen_set = set(train_clicks.keys())
    b = bench_norm.with_columns(
        pl.col("nq").is_in(list(seen_set)).alias("seen"),
        (pl.col("search_infm_params_text").fill_null("") == "").alias("empty_params"),
        pl.col("search_query").str.len_chars().alias("chars"),
    )
    med = b["chars"].median()
    slices = {
        "delivery": b.filter(pl.col("search_is_delivery_search") == 1),
        "empty_params": b.filter(pl.col("empty_params")),
        "has_params": b.filter(~pl.col("empty_params")),
        "short(q<=median)": b.filter(pl.col("chars") <= med),
        "long(q>median)": b.filter(pl.col("chars") > med),
        "cat=0": b.filter(pl.col("search_category") == 0),
        "cat=114": b.filter(pl.col("search_category") == 114),
    }
    print(f"{'срез':<22}{'n':>7}{'seen':>10}")
    for name, sub in slices.items():
        if len(sub):
            print(f"{name:<22}{len(sub):>7}{float(sub['seen'].mean()):>10.4f}")
    print(f"{'ALL':<22}{len(b):>7}{float(b['seen'].mean()):>10.4f}")


# --------------------------------------------------------------------------------------
# Раздел 3. Разложение потерь на eval
# --------------------------------------------------------------------------------------
def build_eval_slice(train: pl.DataFrame, indexed: set) -> pl.DataFrame:
    """Детерминированный eval-срез (идентичен логике main.py после фикса)."""
    uq = (
        train.filter(pl.col("item_id").is_in(list(indexed)))
        .group_by(GROUP_KEYS)
        .agg(pl.col("item_id").alias("relevant_items"))
        .sort(GROUP_KEYS)
    )
    return uq.sample(n=min(N_EVAL, len(uq)), seed=SEED)


def fetch_ranked_pools(searcher: Searcher, embedder, embed_cache, rows: list[dict]) -> list[dict]:
    """Возвращает для каждого запроса пул и RRF-ранжирование по production-фильтрам."""
    from qdrant_client.models import QueryRequest

    results = []
    chunk = 100
    for start in tqdm(range(0, len(rows), chunk), desc="eval search"):
        batch = rows[start : start + chunk]
        texts = [f"{r['search_query'] or ''} {r['search_infm_params_text'] or ''}".strip() for r in batch]
        keys = [md5(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in embed_cache]
        if missing:
            new_vecs = embedder.encode([texts[i] for i in missing], batch_size=256, show_progress_bar=False)
            for i, v in zip(missing, new_vecs):
                embed_cache[keys[i]] = v
        vectors = np.stack([embed_cache[k] for k in keys])

        parsed = [extract_special_params(parse_params(r["search_infm_params_text"] or "", KEY_REGEX)) for r in batch]

        es_searches = []
        for i, r in enumerate(batch):
            body = searcher.build_es_query(
                r["search_query"] or "",
                parsed[i][0],
                parsed[i][1],
                location_id=r["search_location_id"],
                category_id=r["search_category"],
                is_delivery=bool(r["search_is_delivery_search"] or 0),
                size=CANDIDATE_SIZE,
            )
            es_searches += [{"index": searcher.es_indexer.index_name}, body]
        es_resp = searcher.es_indexer.client.msearch(searches=es_searches)

        qd_reqs = []
        for i, r in enumerate(batch):
            loc = r["search_location_id"] if (not bool(r["search_is_delivery_search"] or 0)) else None
            q_filter = searcher.build_qdrant_filter(
                parsed[i][0], parsed[i][1], location_id=loc, category_id=r["search_category"]
            )
            qd_reqs.append(QueryRequest(query=vectors[i].tolist(), filter=q_filter, limit=CANDIDATE_SIZE, with_payload=["item_id"]))
        qd_resp = searcher.qdrant_indexer.client.query_batch_points(
            collection_name=searcher.qdrant_indexer.collection_name, requests=qd_reqs
        )

        for i, r in enumerate(batch):
            es_hits = [
                (h["_id"], float(h.get("_score") or 0.0))
                for h in es_resp["responses"][i].get("hits", {}).get("hits", [])
            ]
            qd_hits = [
                (p.payload["item_id"], float(p.score or 0.0))
                for p in qd_resp[i].points
                if p.payload and "item_id" in p.payload
            ]
            # полное RRF-ранжирование, а не только топ-50
            scores: dict[str, float] = {}
            for rl in (es_hits, qd_hits):
                for rank, (iid, _) in enumerate(rl):
                    scores[iid] = scores.get(iid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            pre_sort_top50 = ranked[:50]
            # воспроизводим продовый price-sort финального топ-50 (см. search_batch)
            if parsed[i][1].sort_order and ranked:
                is_desc = parsed[i][1].sort_order == "desc"
                head = sorted(
                    pre_sort_top50,
                    key=lambda x: searcher.item_prices.get(x, 0.0),
                    reverse=is_desc,
                )
                ranked = head + ranked[50:]
            results.append(
                {
                    "es_ids": {h[0] for h in es_hits},
                    "qd_ids": {h[0] for h in qd_hits},
                    "ranked": ranked,
                    "pre_sort_top50": pre_sort_top50,
                    "special": parsed[i][1],
                }
            )
    return results


def section3(searcher: Searcher, embedder, embed_cache, train: pl.DataFrame, items: pl.DataFrame, indexed: set) -> dict:
    section("РАЗДЕЛ 3. РАЗЛОЖЕНИЕ ПОТЕРЬ НА eval")

    eval_df = build_eval_slice(train, indexed)
    rows = eval_df.to_dicts()
    print(f"eval-срез (детерминированный): queries={len(rows)}, positives(дедуп)={sum(len(set(r['relevant_items'])) for r in rows)}")

    pools = fetch_ranked_pools(searcher, embedder, embed_cache, rows)

    # Перегенерируем артефакты eval детерминированно (аналог исправленного main.py).
    eval_df.with_columns(
        pl.Series("query_id", [f"train_q_{i}" for i in range(len(eval_df))], dtype=pl.Utf8)
    ).write_parquet(OUTPUT_DIR / "train_eval_queries.parquet")
    with open(OUTPUT_DIR / "train_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer"])
        for i, pool in enumerate(pools):
            writer.writerow([f"train_q_{i}", " ".join(pool["ranked"][:50])])
    print("  записаны output/train_eval_queries.parquet и output/train_predictions.csv (детерминированно)")

    item_loc = dict(zip(items["item_id"].to_list(), items["item_location_id"].to_list()))
    item_lat = dict(zip(items["item_id"].to_list(), items["item_latitude"].to_list()))
    item_lon = dict(zip(items["item_id"].to_list(), items["item_longitude"].to_list()))

    counts = Counter()
    per_query = []
    pos_records = []
    for qi, (r, pool) in enumerate(zip(rows, pools)):
        rel = set(r["relevant_items"])
        ranked = pool["ranked"]
        top50 = set(ranked[:50])
        pooled = pool["es_ids"] | pool["qd_ids"]
        qloc = str(r["search_location_id"])
        deliv = bool(r["search_is_delivery_search"] or 0)
        qcat = str(r["search_category"])

        counts["total"] += len(rel)

        for item in rel:
            loc_mismatch = (not deliv) and qloc not in ("0", "") and str(item_loc.get(item)) != qloc
            in_pool = item in pooled
            in_top = item in top50
            if in_top:
                cat = "found"
            elif loc_mismatch:
                # правило из ТЗ: позитив есть в корпусе, но location_id != запроса
                cat = "filtered_location"
            elif in_pool:
                cat = "lost_in_rrf"
            else:
                cat = "not_found_anywhere"
            counts[cat] += 1
            counts["loc_mismatch_all"] += int(loc_mismatch)
            counts["loc_mismatch_absent_pool"] += int(loc_mismatch and not in_pool)
            counts["loc_mismatch_found_top50"] += int(loc_mismatch and in_top)
            counts["loc_mismatch_in_pool_not_top"] += int(loc_mismatch and in_pool and not in_top)
            if loc_mismatch:
                counts["locmism_in_es"] += int(item in pool["es_ids"])
                counts["locmism_in_qd"] += int(item in pool["qd_ids"])
            pos_records.append(
                {
                    "query_id": f"train_q_{qi}",
                    "item_id": item,
                    "category": cat,
                    "q_loc": qloc,
                    "is_delivery": deliv,
                    "empty_params": (r["search_infm_params_text"] or "") == "",
                    "q_chars": len(r["search_query"] or ""),
                    "q_cat": qcat,
                    "loc_mismatch": loc_mismatch,
                    "item_loc": str(item_loc.get(item)),
                    "item_lat": item_lat.get(item),
                    "item_lon": item_lon.get(item),
                    "has_sort": pool["special"].sort_order is not None,
                }
            )

        # recall@N
        rec_n = {}
        for n in (50, 100, 200, 400, 800):
            rec_n[n] = len(rel & set(ranked[:n])) / len(rel) if rel else 0.0
        per_query.append({"n_rel": len(rel), "recall": rec_n, "has_sort": pool["special"].sort_order})

    print("\n3.1 Гранулярный error analysis (по позитивам, дедуп):")
    total = counts["total"]
    for cat in ("found", "filtered_location", "lost_in_rrf", "not_found_anywhere"):
        print(f"  {cat:<22} {counts[cat]:>6}  {counts[cat]/total*100:>6.2f}%")
    print(f"  {'ИТОГО':<22} {total:>6}")
    print(f"  [справка] location_id != запроса (не delivery), всего: {counts['loc_mismatch_all']} "
          f"({counts['loc_mismatch_all']/total*100:.2f}%): found_top50={counts['loc_mismatch_found_top50']}, "
          f"в пуле но не топ-50={counts['loc_mismatch_in_pool_not_top']}, отсутствуют в пуле={counts['loc_mismatch_absent_pool']}")
    print(f"  [каналы] из них найдены в ES-пуле: {counts['locmism_in_es']} "
          f"({counts['locmism_in_es']/max(counts['loc_mismatch_all'],1):.4f}), в Qdrant-пуле: {counts['locmism_in_qd']} "
          f"({counts['locmism_in_qd']/max(counts['loc_mismatch_all'],1):.4f})")
    print("  -> Qdrant жёстко фильтрует по location_id, ES только бустит: ES-канал спасает часть локационных позитивов.")
    print("  (заявлено в задаче: found=1873 79.1%, filtered_location=368 15.5%, lost_in_rrf=124 5.2%, not_found_anywhere=2)")

    # 3.2 срезы
    section("3.2 Потери по срезам")

    pos_df = pl.DataFrame(pos_records)
    base_recall = float(np.mean([pq["recall"][50] for pq in per_query]))
    print(f"средний recall@50 = {base_recall:.4f}\n")
    print(f"{'срез':<26}{'n_q':>6}{'recall@50':>12}")

    loc_sizes = {
        r["item_location_id"]: r["len"]
        for r in items.group_by("item_location_id").agg(pl.len().alias("len")).to_dicts()
    }
    q_meta = []
    for r, pool in zip(rows, pools):
        q_meta.append(
            {
                "empty": (r["search_infm_params_text"] or "") == "",
                "q_chars": len(r["search_query"] or ""),
                "q_cat": str(r["search_category"]),
                "is_delivery": bool(r["search_is_delivery_search"] or 0),
                "has_sort": pool["special"].sort_order is not None,
                "loc_pool": loc_sizes.get(r["search_location_id"], 0),
                "recall": None,
            }
        )
    for pq, qm in zip(per_query, q_meta):
        qm["recall"] = pq["recall"][50]
        qm["n_rel"] = pq["n_rel"]
        qm["recall_n"] = pq["recall"]

    med_chars = float(np.median([q["q_chars"] for q in q_meta]))
    slice_defs = {
        "delivery=True": lambda q: q["is_delivery"],
        "empty params": lambda q: q["empty"],
        "has params": lambda q: not q["empty"],
        f"short(q<={med_chars:.0f})": lambda q: q["q_chars"] <= med_chars,
        f"long(q>{med_chars:.0f})": lambda q: q["q_chars"] > med_chars,
        "cat=114": lambda q: q["q_cat"] == "114",
        "cat=0": lambda q: q["q_cat"] == "0",
        "loc_pool=0": lambda q: q["loc_pool"] == 0,
        "loc_pool 1-49": lambda q: 0 < q["loc_pool"] < 50,
        "loc_pool 50-199": lambda q: 50 <= q["loc_pool"] < 200,
        "loc_pool>=200": lambda q: q["loc_pool"] >= 200,
        "n_positives>1": lambda q: q["n_rel"] > 1,
        "n_positives=1": lambda q: q["n_rel"] == 1,
        "has_sort": lambda q: q["has_sort"],
    }
    for name, pred in slice_defs.items():
        vals = [q["recall"] for q in q_meta if pred(q)]
        if vals:
            print(f"{name:<26}{len(vals):>6}{np.mean(vals):>12.4f}")
    print("\nпояснение: cat вырождена (все 114, кроме cat=0); loc_pool — число объявлений в локации запроса.")

    # 3.3 recall@N
    section("3.3 Recall пула на глубине (RRF)")
    print(f"{'N':>6}{'recall@N':>12}")
    for n in (50, 100, 200, 400, 800):
        vals = [pq["recall"][n] for pq in per_query]
        print(f"{n:>6}{np.mean(vals):>12.4f}")
    r50 = np.mean([pq["recall"][50] for pq in per_query])
    r400 = np.mean([pq["recall"][400] for pq in per_query])
    print(f"\nrecall(400) - recall(50) = {(r400 - r50)*100:.2f} п.п.")

    # 3.4 price-sort
    section("3.4 Подсрез price-sort")
    n_sort = sum(1 for q in q_meta if q["has_sort"])
    print(f"запросов с 'Сортировка для URL': {n_sort} / {len(q_meta)} ({n_sort/len(q_meta):.4f})")
    for name, sel in [("all", lambda q: True), ("sort", lambda q: q["has_sort"]), ("no_sort", lambda q: not q["has_sort"])]:
        vals = [q["recall"] for q in q_meta if sel(q)]
        if vals:
            print(f"  {name:<8} n={len(vals):<6} recall@50={np.mean(vals):.4f}")

    # сколько позитивов теряется из-за пересортировки по цене
    lost_by_sort = 0
    tot_sort_rel = 0
    for r, pool in zip(rows, pools):
        if pool["special"].sort_order is None:
            continue
        rel = set(r["relevant_items"])
        tot_sort_rel += len(rel)
        top50_pre = set(pool["pre_sort_top50"])
        top50_post = set(pool["ranked"][:50])
        if top50_pre != top50_post:
            lost_by_sort += len(rel & top50_pre) - len(rel & top50_post)
    print(f"  позитивов в sort-запросах: {tot_sort_rel}, lost из-за пересортировки: {lost_by_sort}")

    np.save(OUTPUT_DIR / "eval_per_query_recall.npy", np.array([pq["recall"][50] for pq in per_query]))
    return {"rows": rows, "pools": pools, "pos_df": pos_df, "counts": counts, "q_meta": q_meta}


# --------------------------------------------------------------------------------------
# Раздел 4. Локационные потери
# --------------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def section4(pos_df: pl.DataFrame, items: pl.DataFrame, bench: pl.DataFrame) -> None:
    section("РАЗДЕЛ 4. ЛОКАЦИОННЫЕ ПОТЕРИ (filtered_location)")

    # Заявленные 368 = все позитивы с location_id != запроса (не delivery), включая
    # те, что всё же попали в пул/топ-50. Анализируем именно это множество.
    fl = pos_df.filter(pl.col("loc_mismatch"))
    print(f"позитивов с location_id != запроса (не delivery): {fl.height}")
    print(f"  из них category=filtered_location (нет в пуле): {pos_df.filter(pl.col('category') == 'filtered_location').height}")

    # центроиды локаций по объявлениям корпуса
    cent = (
        items.filter(pl.col("item_latitude").is_not_null() & pl.col("item_longitude").is_not_null())
        .group_by("item_location_id")
        .agg(
            pl.col("item_latitude").mean().alias("lat"),
            pl.col("item_longitude").mean().alias("lon"),
            pl.len().alias("n"),
        )
    )
    cent_map = {r["item_location_id"]: (r["lat"], r["lon"], r["n"]) for r in cent.to_dicts()}

    dists = []
    no_centroid = 0
    no_coord = 0
    for rec in fl.to_dicts():
        key = int(rec["q_loc"]) if rec["q_loc"].isdigit() else rec["q_loc"]
        c = cent_map.get(key)
        if not c:
            no_centroid += 1
            continue
        if rec["item_lat"] is None or rec["item_lon"] is None:
            no_coord += 1
            continue
        dists.append(haversine(float(c[0]), float(c[1]), float(rec["item_lat"]), float(rec["item_lon"])))
    dists = np.array(dists)
    print(f"посчитано расстояний: {len(dists)}; нет центроида локации запроса: {no_centroid}; нет координат айтема: {no_coord}")
    print("  (центроид строится только если в локации запроса есть объявления корпуса)")
    if len(dists):
        buckets = [(0, 25), (25, 50), (50, 100), (100, 300), (300, np.inf)]
        print(f"\n{'бакет, км':<16}{'доля':>10}")
        for lo, hi in buckets:
            share = float(np.mean((dists >= lo) & (dists < hi)))
            label = f"{lo}-{hi}" if hi != np.inf else f">{lo}"
            print(f"{label:<16}{share:>10.4f}")
        print(f"\nмедиана={np.median(dists):.1f} км, p90={np.percentile(dists, 90):.1f} км")
        print(f"4.2 прокси 'тот же регион' (<100 км): {float(np.mean(dists < 100)):.4f}")

    # 4.3 признаки удалённой услуги
    kw = re.compile(r"(?i)онлайн|дистанц|выезд|доставк|удал[её]нн|по\s+всей|из\s+любого|под\s+ключ")
    bench_text = [
        (r["search_query"] or "") + " " + (r["search_infm_params_text"] or "")
        for r in bench.to_dicts()
    ]
    n_kw = sum(1 for t in bench_text if kw.search(t))
    print(f"\n4.3 benchmark-запросов с признаками удалённой услуги: {n_kw}/{len(bench_text)} ({n_kw/len(bench_text):.4f})")
    txt = dict(zip(items["item_id"].to_list(), items["text_for_embed"].to_list()))
    if fl.height:
        pos_kw = 0
        n_with_text = 0
        for rec in fl.to_dicts():
            t = txt.get(rec["item_id"])
            if t is None:
                continue
            n_with_text += 1
            if kw.search(t):
                pos_kw += 1
        print(f"4.3 filtered_location позитивов с признаками удалённой услуги: {pos_kw}/{n_with_text} ({pos_kw/max(n_with_text,1):.4f})")
    print("4.3 прим.: в benchmark special params пусты (sort/rating/words = 0%), поэтому 4.3 только по тексту.")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    print("Загрузка данных...")
    train = pl.read_parquet(DATA_DIR / "train.parquet")
    bench = pl.read_parquet(DATA_DIR / "benchmark_queries.parquet")
    items = pl.read_parquet(DATA_DIR / "items_parsed.parquet")
    indexed = set(items["item_id"].to_list())

    s1 = section1(train, bench)
    section2(train, bench, s1["bi_ids"], s1["train_clicks"])

    # сервисы
    es = ESIndexer()
    qd = QdrantIndexer()
    assert es.check_health(), "ES недоступен"
    assert qd.check_health(), "Qdrant недоступен"
    embedder = TextEmbedder(str((BASE_DIR / "finetuned_rubert").absolute()))
    prices = dict(zip(items["item_id"].to_list(), items["item_price"].fill_null(0.0).to_list()))
    searcher = Searcher(es, qd, embedder, item_prices=prices)
    embed_cache = load_query_embed_cache()

    s3 = section3(searcher, embedder, embed_cache, train, items, indexed)
    section4(s3["pos_df"], items, bench)


if __name__ == "__main__":
    main()