"""Utilities for producing bounded, source-contiguous chunk ranges."""

from typing import Iterator


def split_source_ranges(
        source: str, start: int, end: int, maximum: int
        ) -> Iterator[tuple[int, int]]:
    """Yield line-aware ranges no larger than ``maximum`` characters."""
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
