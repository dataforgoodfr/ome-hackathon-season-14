from sentence_transformers import SentenceTransformer, util

from placeholder.interface import TextWithBoundaries

# model = SentenceTransformer("dangvantuan/french-document-embedding", trust_remote_code=True)
model = SentenceTransformer("intfloat/multilingual-e5-base", trust_remote_code=True)

def segment_by_embedding(text, window_size=50, step=20, threshold=0.90) -> TextWithBoundaries:
    """Segment text based on window size and similarity threshold."""
    words = text.split()
    num_words = len(words)

    # 1. Create sliding windows
    # i is the starting word index for each window
    window_starts = list(range(0, max(1, num_words - window_size + 1), step))

    chunks = []
    for start in window_starts:
        end = min(start + window_size, num_words)
        chunks.append(" ".join(words[start:end]))
    if not chunks:
        return TextWithBoundaries(chunks=[], index_boundaries=[0])

    # 2. Get Embeddings
    embeddings = model.encode(chunks)

    # 3. Detect Boundaries
    # We store the word index where the "break" happens
    # If window i and i+1 are different, the break is at the start of window i+1
    word_boundaries = [0]

    for i in range(len(embeddings) - 1):
        similarity = util.cos_sim(embeddings[i], embeddings[i+1])

        if similarity < threshold:
            # Conflict resolution: map the window change back to the word index
            # The "new" context starts at the beginning of the next window
            boundary_word_idx = window_starts[i+1]
            word_boundaries.append(boundary_word_idx)

    # Remove duplicates if multiple overlaps trigger at the same spot
    word_boundaries = sorted(list(set(word_boundaries)))

    return TextWithBoundaries(
        text=text,
        words=words,
        index_boundaries=word_boundaries, # These are word-level indices
    )


def merge_chunks_with_boundaries(text_with_boundaries: TextWithBoundaries) -> list[str]:
    """Merge chunks based on boundaries."""
    segmented_texts = []

    last_idx = 0
    for index, boundary_idx in enumerate(text_with_boundaries.index_boundaries):
        if index != 0:
            segmented_texts.append(" ".join(text_with_boundaries.words[last_idx: boundary_idx + 1]))
            last_idx = boundary_idx
    segmented_texts.append(" ".join(text_with_boundaries.words[last_idx+1:]))

    return segmented_texts
