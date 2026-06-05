from __future__ import annotations


class BaselineModel:
    def fit(self, examples: object) -> "BaselineModel":
        return self

    def predict(self, task: object) -> object:
        return task
