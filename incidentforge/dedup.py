"""Semantic deduplication for IncidentForge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer

from .models import Incident


@dataclass
class DedupConfig:
    """Configuration for deduplication."""

    similarity_threshold: float = 0.92  # Cosine similarity threshold
    minhash_threshold: float = 0.85  # Jaccard similarity threshold
    num_perm: int = 128  # MinHash permutations
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_embeddings: bool = True
    use_minhash: bool = True


class EmbeddingDeduplicator:
    """Deduplication using sentence embeddings and cosine similarity."""

    def __init__(self, config: DedupConfig):
        self.config = config
        self._model: SentenceTransformer | None = None
        self._embeddings: dict[str, np.ndarray] = {}  # incident_id -> embedding
        self._texts: dict[str, str] = {}  # incident_id -> text for embedding

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def _get_incident_text(self, incident: Incident) -> str:
        """Get representative text for incident embedding."""
        parts = [
            incident.service,
            incident.failure_category.value,
            incident.root_cause_analysis,
            " ".join(e.event for e in incident.timeline),
            " ".join(item.action for item in incident.action_items),
        ]
        return " | ".join(parts)

    def add(self, incident: Incident) -> None:
        """Add incident to deduplication index."""
        if not self.config.use_embeddings:
            return

        text = self._get_incident_text(incident)
        self._texts[incident.id] = text
        embedding = self.model.encode(text, normalize_embeddings=True)
        self._embeddings[incident.id] = embedding

    def is_duplicate(self, incident: Incident) -> tuple[bool, str | None, float]:
        """
        Check if incident is a duplicate of any existing.

        Returns:
            Tuple of (is_duplicate, duplicate_id, similarity_score).
        """
        if not self.config.use_embeddings or not self._embeddings:
            return False, None, 0.0

        text = self._get_incident_text(incident)
        embedding = self.model.encode(text, normalize_embeddings=True)

        max_similarity = 0.0
        duplicate_id = None

        for existing_id, existing_embedding in self._embeddings.items():
            similarity = float(np.dot(embedding, existing_embedding))
            if similarity > max_similarity:
                max_similarity = similarity
                duplicate_id = existing_id

        is_dup = max_similarity >= self.config.similarity_threshold
        return is_dup, duplicate_id, max_similarity

    def get_similar(self, incident: Incident, top_k: int = 5) -> list[tuple[str, float]]:
        """Get top-k most similar incidents."""
        if not self.config.use_embeddings or not self._embeddings:
            return []

        text = self._get_incident_text(incident)
        embedding = self.model.encode(text, normalize_embeddings=True)

        similarities = []
        for existing_id, existing_embedding in self._embeddings.items():
            similarity = float(np.dot(embedding, existing_embedding))
            similarities.append((existing_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class MinHashDeduplicator:
    """Deduplication using MinHash LSH for approximate Jaccard similarity."""

    def __init__(self, config: DedupConfig):
        self.config = config
        self._lsh: MinHashLSH | None = None
        self._minhashes: dict[str, MinHash] = {}

    def _get_minhash(self, incident: Incident) -> MinHash:
        """Create MinHash for incident."""
        m = MinHash(num_perm=self.config.num_perm)

        # Create shingles from incident text
        text = self._get_incident_text(incident)
        shingles = self._get_shingles(text)

        for shingle in shingles:
            m.update(shingle.encode("utf-8"))

        return m

    def _get_shingles(self, text: str, k: int = 5) -> set[str]:
        """Generate k-shingles from text."""
        text = text.lower()
        words = text.split()
        shingles = set()

        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i : i + k])
            shingles.add(shingle)

        return shingles

    def _get_incident_text(self, incident: Incident) -> str:
        """Get representative text for MinHash."""
        parts = [
            incident.service,
            incident.failure_category.value,
            incident.root_cause_analysis,
            " ".join(e.event for e in incident.timeline),
            " ".join(item.action for item in incident.action_items),
        ]
        return " | ".join(parts)

    def add(self, incident: Incident) -> None:
        """Add incident to LSH index."""
        if not self.config.use_minhash:
            return

        if self._lsh is None:
            self._lsh = MinHashLSH(
                threshold=self.config.minhash_threshold,
                num_perm=self.config.num_perm,
            )

        minhash = self._get_minhash(incident)
        self._minhashes[incident.id] = minhash
        self._lsh.insert(incident.id, minhash)

    def is_duplicate(self, incident: Incident) -> tuple[bool, str | None, float]:
        """
        Check if incident is a duplicate using MinHash LSH.

        Returns:
            Tuple of (is_duplicate, duplicate_id, estimated_jaccard).
        """
        if not self.config.use_minhash or self._lsh is None:
            return False, None, 0.0

        minhash = self._get_minhash(incident)
        candidates = self._lsh.query(minhash)

        if not candidates:
            return False, None, 0.0

        # Compute actual Jaccard with candidates
        max_jaccard = 0.0
        duplicate_id = None

        for candidate_id in candidates:
            if candidate_id == incident.id:
                continue
            candidate_minhash = self._minhashes.get(candidate_id)
            if candidate_minhash:
                jaccard = minhash.jaccard(candidate_minhash)
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    duplicate_id = candidate_id

        is_dup = max_jaccard >= self.config.minhash_threshold
        return is_dup, duplicate_id, max_jaccard


class DiversityController:
    """Controls diversity of generated incidents."""

    def __init__(self, config: DedupConfig):
        self.config = config
        self.embedding_dedup = EmbeddingDeduplicator(config)
        self.minhash_dedup = MinHashDeduplicator(config)
        self.category_counts: dict[str, int] = {}
        self.total_count = 0

    @classmethod
    def from_incidents(
        cls, incidents: list[Incident], config: DedupConfig | None = None
    ) -> DiversityController:
        """Create controller from existing incidents."""
        if config is None:
            config = DedupConfig()
        controller = cls(config)
        for incident in incidents:
            controller.add(incident)
        return controller

    def add(self, incident: Incident) -> None:
        """Add incident to diversity tracking."""
        self.embedding_dedup.add(incident)
        self.minhash_dedup.add(incident)

        cat = incident.failure_category.value
        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
        self.total_count += 1

    def check_duplicate(self, incident: Incident) -> tuple[bool, str | None, float]:
        """
        Check if incident is duplicate using both methods.

        Returns:
            Tuple of (is_duplicate, duplicate_id, max_similarity).
        """
        # Check embeddings
        emb_dup, emb_id, emb_sim = self.embedding_dedup.is_duplicate(incident)
        if emb_dup:
            return True, emb_id, emb_sim

        # Check MinHash
        mh_dup, mh_id, mh_sim = self.minhash_dedup.is_duplicate(incident)
        if mh_dup:
            return True, mh_id, mh_sim

        return False, None, max(emb_sim, mh_sim)

    def check_category_balance(self, incident: Incident, max_category_pct: float = 0.20) -> bool:
        """
        Check if adding this incident would exceed category balance.

        Args:
            incident: Incident to check.
            max_category_pct: Maximum percentage for any category (default 20%).

        Returns:
            True if adding would maintain balance, False otherwise.
        """
        if self.total_count == 0:
            return True

        cat = incident.failure_category.value
        current_count = self.category_counts.get(cat, 0)
        new_count = current_count + 1
        new_total = self.total_count + 1

        return (new_count / new_total) <= max_category_pct

    def get_category_distribution(self) -> dict[str, float]:
        """Get current category distribution as percentages."""
        if self.total_count == 0:
            return {}

        return {cat: count / self.total_count for cat, count in self.category_counts.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "total_incidents": self.total_count,
            "unique_categories": len(self.category_counts),
            "category_distribution": self.get_category_distribution(),
            "embedding_index_size": len(self.embedding_dedup._embeddings),
            "minhash_index_size": len(self.minhash_dedup._minhashes),
        }


def create_dedup_controller(config: DedupConfig | None = None) -> DiversityController:
    """Create a diversity controller with default or custom config."""
    if config is None:
        config = DedupConfig()
    return DiversityController(config)
