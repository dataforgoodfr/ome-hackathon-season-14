from dataclasses import dataclass


@dataclass
class ChunksWithBoundaries:
    chunks: list[str]
    index_boundaries: list[int]


@dataclass
class Segment:
    content: str
    word_count: int
    category: str | None


@dataclass
class TextAnalysis:
    input_text: str
    initial_word_count: int
    nb_duplicates: int
    final_word_count: int
    segments: list[Segment]
