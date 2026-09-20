"""Модуль эмбеддера с дефолтным cointegrated/rubert-tiny2 с автоматическим выбором ускорителя."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"


def _read_pooling_mode(model_name: str) -> str:
    """Читает режим пулинга из конфига SentenceTransformer-чекпоинта."""
    config_path = Path(model_name) / "1_Pooling" / "config.json"
    if not config_path.exists():
        return "mean"
    try:
        with open(config_path, encoding="utf-8") as f:
            mode = json.load(f).get("pooling_mode", "mean")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read pooling config {config_path}: {e}")
        return "mean"
    if mode not in ("cls", "mean", "last"):
        logger.warning(f"Unknown pooling_mode '{mode}', falling back to 'mean'")
        return "mean"
    return mode


class TextEmbedder:
    """Эмбеддер на базе rubert-tiny2 (либо кастомной модели) с пулингом из конфига и L2 нормализацией."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.pooling_mode = _read_pooling_mode(model_name)
        logger.info(f"Using pooling_mode='{self.pooling_mode}' for {model_name}")
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(requested_device: str | None) -> torch.device:
        if requested_device:
            return torch.device(requested_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 512,
        show_progress_bar: bool = False,
        max_length: int = 128,
    ) -> np.ndarray:
        """Кодирует последовательность текстов в L2-нормализованные 312-dim векторы-эмбеддинги."""
        all_embeddings: list[np.ndarray] = []
        n_samples = len(texts)

        iterator = range(0, n_samples, batch_size)
        if show_progress_bar:
            iterator = tqdm(
                iterator,
                desc="Generating embeddings",
                total=(n_samples + batch_size - 1) // batch_size,
            )

        with torch.no_grad():
            for start_idx in iterator:
                batch_texts = list(texts[start_idx : start_idx + batch_size])
                # Убеждаемся, что пустые строки и None обработаны
                batch_texts = [t if t and t.strip() else " " for t in batch_texts]

                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                model_output = self.model(**encoded_input)
                # Первый элемент содержит скрытое состояние (batch, seq, hidden)
                token_embeddings = model_output[0]

                if self.pooling_mode == "cls":
                    embeddings = token_embeddings[:, 0, :]
                elif self.pooling_mode == "last":
                    input_mask_expanded = (
                        encoded_input["attention_mask"]
                        .unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    # последний непаддированный токен
                    lengths = input_mask_expanded.sum(1).long() - 1
                    embeddings = token_embeddings[
                        torch.arange(token_embeddings.size(0)), lengths
                    ]
                else:
                    # mean pooling с attention mask
                    input_mask_expanded = (
                        encoded_input["attention_mask"]
                        .unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    sum_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, 1
                    )
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                # Нормализация эмбеддингов до единиц измерения (косинусная близость)
                normalized = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

        if not all_embeddings:
            return np.empty((0, 312), dtype=np.float32)

        return np.vstack(all_embeddings)
