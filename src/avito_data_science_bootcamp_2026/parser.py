"""Parser and preprocessor for *_infm_params_text and items datasets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Базовый список ключей из analytics/1_infm_params_aggregation.py + 'Сортировка для URL' и 'Слова в описании'
RAW_KEYS = [
    "Сортировка для URL",
    "Слова в описании",
    "Вид услуги",
    "Тип услуги",
    "Тип товара",
    "Вид товара",
    "Срочная услуга (мультистатус)",
    "Кто оказывает услуги",
    "Рейтинг пользователя",
    "Тип объявления",
    "Сфера деятельности",
    "График работы, дни недели",
    "Марка",
    "Состояние",
    "Вид техники",
    "Открытие в сегменте авито для бизнеса",
    "Марка авто",
    "График работы v2",
    "График работы",
    "Где вы оказываете услуги",
    "Ваши клиенты",
    "Специальность или сфера",
    "Онлайн-запись",
    "Тип автосервиса",
    "Тип услуги автосервиса",
    "Работа по договору",
    "Гарантия",
    "Предмет или специальность",
]

SPECIAL_KEYS = {
    "Сортировка для URL",
    "Рейтинг пользователя",
    "Слова в описании",
}


@dataclass
class SpecialParams:
    """Special parameters extracted from search query."""

    sort_order: str | None = None  # "asc", "desc", or None
    min_rating: float | None = None  # e.g., 4.0 or None
    description_words: list[str] | None = None  # raw words from 'Слова в описании'


def build_key_regex(keys: list[str]) -> re.Pattern:
    """Build case-sensitive regex matching any of the title-case keys."""
    keys_sorted = sorted(set(keys), key=len, reverse=True)
    pattern = "|".join(re.escape(k) for k in keys_sorted)
    return re.compile(rf"\b({pattern})")


KEY_REGEX = build_key_regex(RAW_KEYS)


def _is_boundary_token(word: str) -> bool:
    """Check if token marks the end of a parameter value."""
    if not word:
        return False
    first = word[0]

    # Markers [поиск], {...} are boundaries
    if first in "[{":
        return True

    # Non-uppercase first character is not a boundary
    if not first.isupper():
        return False

    # Acronyms: all uppercase letters, length <= 4 (ТО, BMW, ИП, ГБО, МКПП, ХВС)
    letters = [c for c in word if c.isalpha()]

    return not (letters and all(c.isupper() for c in letters) and len(letters) <= 4)


def _find_value_bounds(segment: str) -> tuple[int, int]:
    """Find start and end of value in segment."""
    if not segment.strip():
        return 0, 0

    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\s]+", segment)]
    if not tokens:
        return 0, len(segment)

    start = tokens[0][0]
    for i in range(1, len(tokens)):
        if _is_boundary_token(tokens[i][2]):
            return start, tokens[i][0]
    return start, len(segment)


def parse_params(text: str, key_regex: re.Pattern = KEY_REGEX) -> dict[str, list[str]]:
    """Key-value parser extracting parameter values while respecting title-case bounds."""
    if not text:
        return {}

    matches = list(key_regex.finditer(text))
    if not matches:
        return {}

    result: dict[str, list[str]] = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        seg_start = m.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[seg_start:seg_end]

        v_start, v_end = _find_value_bounds(segment)
        value = segment[v_start:v_end].strip(" ,;.")

        result.setdefault(key, [])
        if value:
            result[key].append(value)
    return result


def to_slug(key: str) -> str:
    """Convert title-case key to slug: lower, spaces to _, remove non-alphanumeric/underscore."""
    lowered = key.lower().strip()
    underscored = re.sub(r"[\s\-]+", "_", lowered)
    cleaned = re.sub(r"[^\w_]", "", underscored)
    return cleaned.strip("_")


def extract_special_params(
    raw_params: dict[str, list[str]],
) -> tuple[dict[str, list[str]], SpecialParams]:
    """Separate standard parameters from the three special keys."""
    special = SpecialParams()
    regular_params: dict[str, list[str]] = {}

    for k, vals in raw_params.items():
        if k == "Сортировка для URL":
            val_text = " ".join(vals).lower()
            if "дешев" in val_text:
                special.sort_order = "asc"
            elif "дорог" in val_text:
                special.sort_order = "desc"
        elif k == "Рейтинг пользователя":
            val_text = " ".join(vals)
            match = re.search(r"(\d+(?:[.,]\d+)?)", val_text)
            if match:
                special.min_rating = float(match.group(1).replace(",", "."))
        elif k == "Слова в описании":
            words: list[str] = []
            for v in vals:
                for w in v.split():
                    w_clean = w.strip(" ,;.:\"'()[]{}!?-")
                    if w_clean:
                        words.append(w_clean)
            if words:
                special.description_words = words
        else:
            regular_params[k] = vals

    return regular_params, special


def preprocess_items_df(
    items_df: pl.DataFrame, cache_path: Path | None = None
) -> pl.DataFrame:
    """Parse item_infm_params_text, build params_flat, params_keys, and param_<slug> fields.

    Saves to parquet cache if cache_path is provided.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached preprocessed items from {cache_path}")
        return pl.read_parquet(cache_path)

    logger.info("Preprocessing items dataset: parsing infm_params_text...")
    raw_texts = items_df["item_infm_params_text"].fill_null("").to_list()

    all_parsed: list[dict[str, list[str]]] = []
    for text in raw_texts:
        all_parsed.append(parse_params(text, KEY_REGEX))

    # Identify all slugs from non-special keys
    all_slug_keys: set[str] = set()
    for d in all_parsed:
        for k in d:
            if k not in SPECIAL_KEYS:
                all_slug_keys.add(to_slug(k))

    logger.info(f"Identified {len(all_slug_keys)} distinct param slugs across items")

    params_flat_list: list[str] = []
    params_keys_list: list[list[str]] = []
    slug_columns: dict[str, list[list[str]]] = {slug: [] for slug in all_slug_keys}
    text_for_embed_list: list[str] = []

    titles = items_df["item_title_raw"].fill_null("").to_list()
    descriptions = items_df["item_description_raw"].fill_null("").to_list()

    for idx, d in enumerate(all_parsed):
        # 1. params_keys and param_<slug>
        item_keys: list[str] = []
        flat_parts: list[str] = []
        item_slug_vals: dict[str, list[str]] = {}

        for k, vals in d.items():
            if k in SPECIAL_KEYS or not vals:
                continue
            item_keys.append(k)
            slug = to_slug(k)
            item_slug_vals[slug] = vals
            # Use the natural key (e.g. "Вид услуги") so that params_flat matches
            # the format of search_infm_params_text on the query side — both for
            # BM25 in ES and for the shared embedding space.
            flat_parts.append(f"{k}: {' '.join(vals)}")

        params_flat = " | ".join(flat_parts)
        params_flat_list.append(params_flat)
        params_keys_list.append(item_keys)

        for slug in all_slug_keys:
            slug_columns[slug].append(item_slug_vals.get(slug, []))

        # 2. text_for_embed = title + " " + params_flat + " " + description
        # params_flat goes BEFORE the description snippet: with max_length=128
        # tokens anything at the tail gets truncated, and params are more
        # informative than the description tail. Full description is passed
        # into text_for_embed.
        title = titles[idx]
        desc = descriptions[idx]
        text_for_embed = f"{title} {params_flat} {desc}".strip()
        text_for_embed_list.append(text_for_embed)

    processed_df = items_df.with_columns(
        [
            pl.Series("params_flat", params_flat_list, dtype=pl.Utf8),
            pl.Series("params_keys", params_keys_list, dtype=pl.List(pl.Utf8)),
            pl.Series("text_for_embed", text_for_embed_list, dtype=pl.Utf8),
        ]
        + [
            pl.Series(f"param_{slug}", slug_columns[slug], dtype=pl.List(pl.Utf8))
            for slug in sorted(all_slug_keys)
        ]
    )

    if cache_path:
        logger.info(f"Saving preprocessed items to cache at {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.write_parquet(cache_path)

    return processed_df
