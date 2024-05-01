import pandas as pd
import spacy
from .matcher import Matcher

NLP = spacy.load("es_core_news_lg")


def rbm(input: pd.DataFrame, output: str) -> pd.DataFrame:
    MATCHER = Matcher()
    data = []
    for _, row in input.iterrows():
        respuestas = MATCHER.get_pairs(row["descripcion"])
        data.append(
            {
                "descripcion": row["descripcion"],
                "direccion": respuestas["direccion"],
                "fot": respuestas["fot"],
                "irregular": respuestas["irregular"],
                "medidas": respuestas["medidas"],
                "esquina": respuestas["esquina"],
                "barrio": respuestas["barrio"],
                "frentes": respuestas["frentes"],
                "pileta": respuestas["pileta"],
            }
        )

    df = pd.DataFrame(data, index=None)
    df.set_index("descripcion", inplace=True)
    df.to_csv(output, sep="|")
    return df
