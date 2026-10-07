"""阶段计数。未知保持 null，不用集合差编造丢弃原因。"""

from __future__ import annotations


def stage_count_balanced(
    input_n: int | None,
    created_n: int | None,
    output_n: int | None,
    removed_n: int | None,
) -> bool | None:
    values = (input_n, created_n, output_n, removed_n)
    if any(value is None for value in values):
        return None
    return input_n + created_n == output_n + removed_n
