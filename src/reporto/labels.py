from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Iterable


Label = Literal["mobility_transport", "agriculture_alimentation", "energy", "other"]
LABELS: tuple[Label, ...] = get_args(Label)
LABEL_TO_NUM = {label: i for i, label in enumerate(LABELS)}


def get_numerical_labels(labels: Iterable[Label]) -> list[int]:
    return list(map(LABEL_TO_NUM.__getitem__, labels))
