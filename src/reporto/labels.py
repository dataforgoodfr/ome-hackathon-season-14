from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Iterable


Label = Literal["mobility_transport", "agriculture_alimentation", "energy", "other"]
LABELS: tuple[Label, ...] = get_args(Label)
LABEL_TO_NUM = {label: i for i, label in enumerate(LABELS)}

Label_report = Literal["report", "segment"]
LABELS_reports: tuple[Label_report, ...] = get_args(Label_report)
LABEL_reports_TO_NUM = {label: i for i, label in enumerate(LABELS_reports)}


def get_numerical_labels(labels: Iterable[Label]) -> list[int]:
    return list(map(LABEL_TO_NUM.__getitem__, labels))


def get_numerical_labels_task1(labels: Iterable[Label_report]) -> list[int]:
    return list(map(LABEL_reports_TO_NUM.__getitem__, labels))
