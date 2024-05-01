import json

import pandas as pd
import spacy
from .metric import Metric



NLP = spacy.load("es_core_news_lg")


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    "Sanitize the DataFrame"
    result: pd.DataFrame = df.fillna("")

    result["fot"] = result["fot"].apply(
        lambda x: " ".join(sorted(x.split(". "))).strip()
    )

    return result


class Evaluation:
    """Class to evaluate the results of the extraction model"""

    def __init__(self, gt: pd.DataFrame, results: pd.DataFrame):
        self.gt = sanitize(gt)
        self.results = sanitize(results)
        self.metrics: list[Metric] = [
            Metric(name="direccion"),
            Metric(name="fot"),
            Metric(name="irregular"),
            Metric(name="medidas"),
            Metric(name="esquina"),
            Metric(name="barrio"),
            Metric(name="frentes"),
            Metric(name="pileta"),
        ]

    def evaluate(self) -> list[Metric]:
        "Evaluate the results"
        for actual, expected in zip(
            self.results.itertuples(index=False, name=None),
            self.gt.itertuples(index=False, name=None),
        ):
            for metric in self.metrics:
                actual_value = getattr(actual, metric.name)
                expected_value = getattr(expected, metric.name)

                if actual_value == "" and expected_value == "":
                    metric.tn += 1

                elif self.compare(actual_value, expected_value) == 1:
                    metric.tp += 1

                else:
                    metric.add_error(expected.descripcion, actual_value, expected_value)
                    if actual_value == "" and expected_value != "":
                        metric.fn += 1

                    elif expected_value != actual_value:
                        metric.fp += 1

        return self.metrics

    def __str__(self) -> str:
        "Return the results as a string"
        metrics = [metric.dict() for metric in self.metrics]
        return json.dumps(metrics, ensure_ascii=False)

    def compare(self, actual: str, expected: str) -> float:
        "Compare two strings"
        return NLP(actual.lower()).similarity(NLP(expected.lower()))

    def save(self, path: str):
        "Save the results"
        metrics = [metric.dict() for metric in self.metrics]
        with open(path, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False)
