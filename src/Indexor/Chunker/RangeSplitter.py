"""Utilities for producing bounded, source-contiguous chunk ranges."""

from typing import Iterator


def split_source_ranges(
        source: str, start: int, end: int, maximum: int, overlap: int = 0
        ) -> Iterator[tuple[int, int]]:
    """Yield bounded source ranges with optional character overlap."""
    if maximum <= 0:
        raise ValueError("maximum must be greater than zero")
    if overlap < 0 or overlap >= maximum:
        raise ValueError("overlap must be between zero and maximum - 1")
    if overlap:
        yield from _split_overlapping_ranges(
            source, start, end, maximum, overlap
        )
        return

    yield from _split_line_ranges(source, start, end, maximum)


def _split_line_ranges(
        source: str, start: int, end: int, maximum: int
        ) -> Iterator[tuple[int, int]]:
    """Yield non-overlapping, line-aware ranges."""
    current = start
    for line in source[start:end].splitlines(keepends=True):
        line_end = current + len(line)
        if line_end == current:
            continue
        segment_start = current
        while segment_start < line_end:
            segment_end = min(segment_start + maximum, line_end)
            if segment_end < line_end:
                boundary = segment_end
                while (
                        boundary > segment_start
                        and not source[boundary - 1].isspace()):
                    boundary -= 1
                if boundary > segment_start:
                    segment_end = boundary
            yield segment_start, segment_end
            segment_start = segment_end
        current = line_end

    if current < end:
        yield current, end


def _split_overlapping_ranges(
        source: str, start: int, end: int, maximum: int, overlap: int
        ) -> Iterator[tuple[int, int]]:
    """Yield bounded sliding windows, preferring whitespace boundaries."""
    current = start
    while current < end:
        segment_end = min(current + maximum, end)
        if segment_end < end:
            boundary = segment_end
            while (
                    boundary > current
                    and not source[boundary - 1].isspace()
            ):
                boundary -= 1
            if boundary > current:
                segment_end = boundary
        yield current, segment_end
        if segment_end == end:
            return
        current = max(current + 1, segment_end - overlap)
