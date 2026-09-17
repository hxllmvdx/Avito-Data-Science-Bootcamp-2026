"""Qdrant indexer class managing collection creation, payload indexing, and vector upload."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from tqdm import tqdm

from avito_data_science_bootcamp_2026.embeddings import TextEmbedder

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "avito_items"
VECTOR_DIM = 312


class QdrantIndexer:
    """Manages Qdrant vector database operations for Avito benchmark items."""

    def __init__(
        self,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=self.qdrant_url, timeout=60.0, check_compatibility=False
        )

    def check_health(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {e}")
            return False

    def _load_up(
        self,
        items_df: pl.DataFrame,
        embedder: TextEmbedder,
        batch_size: int = 1000,
        embed_batch_size: int = 512,
        embeddings_cache_path: Any = None,
    ) -> None:
        """Recreate collection, build embeddings, and upload vectors with payload."""
        if not self.check_health():
            raise ConnectionError(
                f"Qdrant is not available at {self.qdrant_url}. Make sure docker-compose services are running."
            )

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

        # Create payload indexes for fast filtering and sorting
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="rating",
            field_schema=PayloadSchemaType.FLOAT,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="price",
            field_schema=PayloadSchemaType.FLOAT,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="location_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        param_slug_cols = [c for c in items_df.columns if c.startswith("param_")]
        n_items = len(items_df)

        # 1. Compute or load embeddings for text_for_embed
        embeddings: np.ndarray | None = None
        if embeddings_cache_path:
            import os
            from pathlib import Path

            cache_p = Path(embeddings_cache_path)
            if cache_p.exists():
                try:
                    loaded = np.load(cache_p)
                    if loaded.shape == (n_items, VECTOR_DIM):
                        embeddings = loaded
                except Exception:
                    embeddings = None

        if embeddings is None:
            texts = items_df["text_for_embed"].fill_null("").to_list()
            embeddings = embedder.encode(
                texts,
                batch_size=embed_batch_size,
                show_progress_bar=True,
            )
            if embeddings_cache_path:
                from pathlib import Path

                cache_p = Path(embeddings_cache_path)
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_p, embeddings)

        # 2. Prepare payload structures
        item_ids = items_df["item_id"].to_list()
        category_ids = (
            items_df["item_category_id"].cast(pl.Utf8).fill_null("").to_list()
        )
        location_ids = (
            items_df["item_location_id"].cast(pl.Utf8).fill_null("").to_list()
        )
        ratings = items_df["item_rating"].fill_null(0.0).to_list()
        prices = items_df["item_price"].fill_null(0.0).to_list()

        slug_data = {col: items_df[col].to_list() for col in param_slug_cols}

        # 3. Upload to Qdrant using integer index IDs (or UUIDs) while storing item_id in payload
        logger.info(
            f"Uploading points to Qdrant collection in batches of {batch_size}..."
        )
        progress = tqdm(total=n_items, desc="Qdrant vector upload")

        for start_idx in range(0, n_items, batch_size):
            end_idx = min(start_idx + batch_size, n_items)
            points_batch: list[PointStruct] = []

            for i in range(start_idx, end_idx):
                payload: dict[str, Any] = {
                    "item_id": item_ids[i],
                    "category_id": category_ids[i],
                    "location_id": location_ids[i],
                    "rating": float(ratings[i]) if ratings[i] is not None else 0.0,
                    "price": float(prices[i]) if prices[i] is not None else 0.0,
                }
                for col in param_slug_cols:
                    val_list = slug_data[col][i]
                    if val_list:
                        payload[col] = val_list

                # Qdrant accepts unsigned 64-bit integer point IDs or UUIDs
                point = PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
                points_batch.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points_batch,
                wait=True,
            )
            progress.update(len(points_batch))

        progress.close()
        logger.info("Qdrant points upload completed")
