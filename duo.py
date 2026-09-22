"""Duo polarity scoring used to select EBI seed passages."""

from __future__ import annotations

import os
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA


class Duo:
    """Project passages onto a one-dimensional synthetic-corpus polarity axis."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int | None = 7,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(embedding_model)
        self.pca = PCA(n_components=1, random_state=random_state)
        self.embeddings: dict[str, float] = {}
        self._set_random_state(random_state)

    @staticmethod
    def _set_random_state(seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def embed_texts(self, texts: Sequence[str], normalize: bool = False) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = np.asarray(embeddings)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    @staticmethod
    def _texts(documents: Mapping[str, Mapping[str, str]]) -> tuple[list[str], list[str]]:
        ids = list(documents)
        return ids, [str(documents[doc_id]["text"]) for doc_id in ids]

    def embed(
        self,
        transform_docs: Mapping[str, Mapping[str, str]],
        fit_docs: Mapping[str, Mapping[str, str]] | None = None,
        spurious_docs: Sequence[Mapping[str, Mapping[str, str]]] = (),
    ) -> dict[str, float]:
        """Fit the axis on ``fit_docs`` and score ``transform_docs``.

        If ``spurious_docs`` is a pair of document maps, the difference-of-means
        direction between them is projected out of the embeddings before fitting
        and transforming, to remove that confounding subspace from the polarity axis.
        """
        fit_docs = fit_docs or transform_docs
        _, fit_texts = self._texts(fit_docs)
        transform_ids, transform_texts = self._texts(transform_docs)

        if len(fit_texts) < 2:
            raise ValueError("Duo requires at least two fit passages")

        subspace = None
        if len(spurious_docs) == 2:
            _, sp1_texts = self._texts(spurious_docs[0])
            _, sp2_texts = self._texts(spurious_docs[1])
            emb1 = self.embed_texts(sp1_texts, normalize=True)
            emb2 = self.embed_texts(sp2_texts, normalize=True)
            subspace = emb1.mean(axis=0) - emb2.mean(axis=0)

        def _remove_subspace(embeddings: np.ndarray) -> np.ndarray:
            if subspace is None:
                return embeddings
            return embeddings - np.multiply(
                np.tile(subspace, (embeddings.shape[0], 1)).T,
                np.dot(embeddings, subspace),
            ).T

        fit_embeddings = _remove_subspace(self.embed_texts(fit_texts))
        self.pca.fit(fit_embeddings)

        transform_embeddings = _remove_subspace(self.embed_texts(transform_texts))
        projected = self.pca.transform(transform_embeddings).ravel()

        self.embeddings = {
            doc_id: float(score) for doc_id, score in zip(transform_ids, projected)
        }
        return self.embeddings.copy()

    def embed_answer(self, answer: str) -> float:
        """Project one generated answer onto the already-fitted polarity axis."""
        if not hasattr(self.pca, "components_"):
            raise RuntimeError("Call embed() before embed_answer()")
        return float(self.pca.transform(self.embed_texts([answer])).ravel()[0])

    def select_extreme(self, polarity: str) -> str:
        """Return the most positive or negative synthetic document ID."""
        if not self.embeddings:
            raise RuntimeError("Call embed() before selecting an extreme")
        polarity = polarity.strip().upper()
        if polarity == "PRO":
            return max(self.embeddings, key=self.embeddings.get)
        if polarity == "CON":
            return min(self.embeddings, key=self.embeddings.get)
        raise ValueError("polarity must be PRO or CON")