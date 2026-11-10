"""CPU-only scalar validation shared by the experimental wrapper and tests."""

MAX_CHUNK_LEN = (1 << 32) - 1


def validate_chunk_len(value):
    value = int(value)
    if not 1 <= value <= MAX_CHUNK_LEN:
        raise RuntimeError("chunk_len must be in [1, 4294967295].")
    return value
