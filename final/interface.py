from dataclasses import dataclass


@dataclass
class ChunksWithBoundaries:
    chunks: list[str]
    index_boundaries: list[int]
