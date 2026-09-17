"""Text embedding module using cointegrated/rubert-tiny2 with hardware acceleration."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"


class TextEmbedder:
    """Embedder using cointegrated/rubert-tiny2 with mean pooling and L2 normalization."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
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
        """Encode a sequence of texts into L2-normalized 312-dim embedding vectors."""
        all_embeddings: list[np.ndarray] = []
        n_samples = len(texts)

        iterator = range(0, n_samples, batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Generating embeddings", total=(n_samples + batch_size - 1) // batch_size)

        with torch.no_grad():
            for start_idx in iterator:
                batch_texts = list(texts[start_idx : start_idx + batch_size])
                # Ensure empty or None strings are handled
                batch_texts = [t if t and t.strip() else " " for t in batch_texts]

                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                model_output = self.model(**encoded_input)
                # Mean pooling with attention mask
                token_embeddings = model_output[0]  # First element contains hidden state
                input_mask_expanded = (
                    encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Normalize embeddings to unit length (cosine distance)
                normalized = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

        if not all_embeddings:
            return np.empty((0, 312), dtype=np.float32)

        return np.vstack(all_embeddings)
