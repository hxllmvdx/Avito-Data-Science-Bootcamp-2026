"""Elasticsearch indexer class managing mapping, indexing, and health checks."""

from __future__ import annotations

import logging
from typing import Any, Generator, Sequence

import polars as pl
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_ES_URL = "http://localhost:9200"
INDEX_NAME = "avito_items"


class ESIndexer:
    """Manages Elasticsearch operations for Avito benchmark items."""

    def __init__(self, es_url: str = DEFAULT_ES_URL, index_name: str = INDEX_NAME) -> None:
        self.es_url = es_url
        self.index_name = index_name
        self.client = Elasticsearch(self.es_url, request_timeout=60)

    def check_health(self) -> bool:
        """Check if Elasticsearch is healthy."""
        try:
            health = self.client.cluster.health()
            status = health.get("status")
            logger.info(f"Elasticsearch cluster status: {status}")
            return status in ("green", "yellow")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch at {self.es_url}: {e}")
            return False

    def get_mapping(self, param_slug_cols: Sequence[str]) -> dict[str, Any]:
        """Generate ES mapping according to task specification."""
        properties: dict[str, Any] = {
            "item_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "russian"},
            "description": {"type": "text", "analyzer": "russian"},
            "params_flat": {"type": "text", "analyzer": "russian"},
            "params_keys": {"type": "keyword"},
            "category_id": {"type": "keyword"},
            "microcat_id": {"type": "keyword"},
            "location_id": {"type": "keyword"},
            "price": {"type": "float"},
            "rating": {"type": "float"},
            "reviews_count": {"type": "integer"},
            "location": {"type": "geo_point", "ignore_malformed": True},
        }

        # Add param_<slug> keyword fields
        for slug_col in param_slug_cols:
            properties[slug_col] = {"type": "keyword"}

        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",  # speed up bulk indexing
            },
            "mappings": {
                "properties": properties,
            },
        }

    def _load_up(self, items_df: pl.DataFrame, chunk_size: int = 2000) -> None:
        """Recreate mapping and index all items."""
        if not self.check_health():
            raise ConnectionError(
                f"Elasticsearch is not available at {self.es_url}. Make sure docker-compose services are running."
            )

        param_slug_cols = [c for c in items_df.columns if c.startswith("param_")]
        logger.info(f"Setting up Elasticsearch index '{self.index_name}' with {len(param_slug_cols)} param fields...")

        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Deleting existing index '{self.index_name}'...")
            self.client.indices.delete(index=self.index_name)

        mapping = self.get_mapping(param_slug_cols)
        self.client.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Index '{self.index_name}' created successfully")

        # Convert dataframe columns to python structures for fast iterator
        item_ids = items_df["item_id"].to_list()
        titles = items_df["item_title_raw"].fill_null("").to_list()
        descriptions = items_df["item_description_raw"].fill_null("").to_list()
        params_flats = items_df["params_flat"].fill_null("").to_list()
        params_keys = items_df["params_keys"].to_list()

        category_ids = items_df["item_category_id"].cast(pl.Utf8).fill_null("").to_list()
        microcat_ids = items_df["item_microcat_id"].cast(pl.Utf8).fill_null("").to_list()
        location_ids = items_df["item_location_id"].cast(pl.Utf8).fill_null("").to_list()

        prices = items_df["item_price"].fill_null(0.0).to_list()
        ratings = items_df["item_rating"].fill_null(0.0).to_list()
        reviews = items_df["item_rating_reviews_count"].fill_null(0).to_list()

        lats = items_df["item_latitude"].to_list()
        lons = items_df["item_longitude"].to_list()

        slug_data = {col: items_df[col].to_list() for col in param_slug_cols}

        n_items = len(items_df)

        def action_generator() -> Generator[dict[str, Any], None, None]:
            for i in range(n_items):
                source: dict[str, Any] = {
                    "item_id": item_ids[i],
                    "title": titles[i],
                    "description": descriptions[i],
                    "params_flat": params_flats[i],
                    "params_keys": params_keys[i] or [],
                    "category_id": category_ids[i],
                    "microcat_id": microcat_ids[i],
                    "location_id": location_ids[i],
                    "price": float(prices[i]) if prices[i] is not None else 0.0,
                    "rating": float(ratings[i]) if ratings[i] is not None else 0.0,
                    "reviews_count": int(reviews[i]) if reviews[i] is not None else 0,
                }

                lat, lon = lats[i], lons[i]
                if lat is not None and lon is not None:
                    source["location"] = {"lat": float(lat), "lon": float(lon)}

                for col in param_slug_cols:
                    val_list = slug_data[col][i]
                    if val_list:
                        source[col] = val_list

                yield {
                    "_index": self.index_name,
                    "_id": item_ids[i],
                    "_source": source,
                }

        logger.info(f"Bulk indexing {n_items} items into Elasticsearch...")
        progress = tqdm(total=n_items, desc="Elasticsearch indexing")
        success_count = 0

        for success, info in streaming_bulk(
            self.client,
            action_generator(),
            chunk_size=chunk_size,
            request_timeout=120,
            raise_on_error=True,
        ):
            if success:
                success_count += 1
                progress.update(1)

        progress.close()
        logger.info(f"Indexing completed: {success_count}/{n_items} indexed")

        # Restore refresh interval and refresh index
        self.client.indices.put_settings(
            index=self.index_name,
            body={"index": {"refresh_interval": "1s"}},
        )
        self.client.indices.refresh(index=self.index_name)
        logger.info("Elasticsearch index refreshed and ready")
