from __future__ import annotations


def exact_match_score(y_true: object, y_pred: object) -> float:
    return float(y_true == y_pred)
