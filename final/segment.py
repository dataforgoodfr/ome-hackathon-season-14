from sentence_transformers import SentenceTransformer, util

from final.interface import ChunksWithBoundaries

model = SentenceTransformer("dangvantuan/french-document-embedding", trust_remote_code=True)


def segment_by_embedding(text, window_size=50, threshold=0.7) -> ChunksWithBoundaries:
    """Segment text based on window size and similarity threshold."""
    words = text.split()
    chunks = [" ".join(words[i:i+window_size]) for i in range(0, len(words), window_size)]
    embeddings = model.encode(chunks)

    index_boundaries = [0]
    for i in range(len(embeddings) - 1):
        similarity = util.cos_sim(embeddings[i], embeddings[i+1])
        if similarity < threshold:
            index_boundaries.append(i + 1)

    return ChunksWithBoundaries(
        chunks=chunks,
        index_boundaries=index_boundaries,
    )


def merge_chunks_with_boundaries(chunks_with_boundaries: ChunksWithBoundaries) -> list[str]:
    """Merge chunks based on boundaries."""
    segmented_texts = []

    # We iterate through the boundaries in pairs: (0, 2), (2, 4), (4, 5), etc.
    # We add the length of the chunks list at the end to catch the final segment.
    extended_boundaries = chunks_with_boundaries.index_boundaries + [len(chunks_with_boundaries.chunks)]

    for start, end in zip(extended_boundaries, extended_boundaries[1:]):
        group = chunks_with_boundaries.chunks[start:end]

        merged_segment = " ".join(group)
        segmented_texts.append(merged_segment)

    return segmented_texts
