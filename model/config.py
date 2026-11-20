"""Central configuration.

Mirrors the `embedding_spaces` registry defined in 001_ella_init.sql so that the
Python pipeline and the database never disagree about which model produces which
dimension for which modality. If you add a row to the SQL registry, add it here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingSpace:
    modality: str
    model_id: str
    dim: int
    column_name: str
    match_threshold: float
    is_active: bool


# Keep this list in sync with the seed rows in 001_ella_init.sql.
EMBEDDING_SPACES: list[EmbeddingSpace] = [
    EmbeddingSpace("nasal_planum", "facebook/dinov2-small", 384, "embedding_384", 0.86, True),
    EmbeddingSpace("cat_face", "facebook/dinov2-small", 384, "embedding_384", 0.86, True),
    # Demo only, non-commercial. Never ship. Kept for benchmarking.
    EmbeddingSpace("nasal_planum", "BVRA/MegaDescriptor-L-384", 1536, "embedding_1536", 0.75, False),
    EmbeddingSpace("cat_face", "BVRA/MegaDescriptor-L-384", 1536, "embedding_1536", 0.75, False),
]

# Native output dimension of each DINOv2 size. DINOv2-giant is 1536-d, which
# lines up with the embedding_1536 column, so a fine-tuned giant is the planned
# commercial launch model (Apache-2.0, unlike MegaDescriptor).
DINOV2_DIMS: dict[str, int] = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
}

# Defaults for the active production path.
DEFAULT_MODALITY = "nasal_planum"
DEFAULT_MODEL_ID = "facebook/dinov2-small"

# Square crop size fed to the backbone after nose detection + preprocessing.
INPUT_SIZE = 224


def space_for(modality: str, model_id: str) -> EmbeddingSpace:
    for s in EMBEDDING_SPACES:
        if s.modality == modality and s.model_id == model_id:
            return s
    raise ValueError(
        f"No embedding space for modality={modality!r} model_id={model_id!r}. "
        f"Add it to EMBEDDING_SPACES and to the SQL registry."
    )
