"Run an AI model for QA, NER and RBM and compute the results"

import argparse

import pandas as pd
from src.helper import descubrir_nuevos
from src.metrics import Evaluation
from src.ner import ner
from src.qa import qa
from src.rbm import rbm

INPUT_WITH_INFERENCES: str = "ground_truth_100.csv"
INPUT_WITHOUT_INFERENCES: str = "ground_truth_100_sin_inferencias.csv"


def main():
    args: argparse.Namespace = parse_args()
    input: pd.DataFrame = process_input(args)

    results: list[pd.DataFrame] = args.script(input)
    results = [process_result(result, args) for result in results]
    for result in results:
        evaluation = Evaluation(input, result)
        evaluation.evaluate()
        print(evaluation)


def process_input(args: argparse.Namespace) -> pd.DataFrame:
    input: str = INPUT_WITH_INFERENCES if args.inferences else INPUT_WITHOUT_INFERENCES
    df: pd.DataFrame = pd.read_csv(input, sep="|")
    return df


def process_result(result: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.inferences:
        clean_result = result.apply(descubrir_nuevos, axis=1)
        assert type(clean_result) == pd.DataFrame
        result = clean_result
    return result


def run_qa(input: pd.DataFrame) -> list[pd.DataFrame]:

    params: list[dict] = [
        {
            "model": "rvargas93/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            "output": "qa/respuestas_rvargas93.csv",
        },
        {
            "model": "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            "output": "qa/respuestas_mrm8488.csv",
        },
        {
            "model": "timpal0l/mdeberta-v3-base-squad2",
            "output": "qa/respuestas_timpal0l.csv",
        },
    ]

    return [qa(input, **param) for param in params]


def run_rbm(input: pd.DataFrame) -> list[pd.DataFrame]:

    params: list[dict] = [
        {
            "output": "rbm/respuestas.csv",
        }
    ]

    return [rbm(input, **param) for param in params]


def run_ner(input: pd.DataFrame) -> list[pd.DataFrame]:

    params: list[dict] = [
        {
            "output": "ner/respuestas.csv",
        }
    ]

    return [ner(input, **param) for param in params]


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--qa",
        action="store_const",
        const={"script": run_qa},
        help="Run QA model",
    )
    group.add_argument(
        "--rbm",
        action="store_const",
        const={"script": run_rbm},
        help="Run RBM model",
    )
    group.add_argument(
        "--ner",
        action="store_const",
        const={"script": run_ner},
        help="Run NER model",
    )

    parser.add_argument(
        "-i",
        "--inferences",
        action=argparse.BooleanOptionalAction,
        help="Run with inferences",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
