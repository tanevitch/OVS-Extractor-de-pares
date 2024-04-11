
from matcher import Matcher
import pandas as pd
import spacy
NLP = spacy.load("es_core_news_lg")


METRICAS = {
        "direccion": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "fot": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "irregular": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "medidas": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "esquina": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "barrio": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "frentes": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        },
        "pileta": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "error": [
                
            ]
        }
}

MATCHER = Matcher()


input = pd.read_csv('ground_truth_100.csv', sep = '|')
input = input.fillna("")
data = []
for index, row in input.iterrows():
    respuestas = MATCHER.get_pairs(row['descripcion'])
    data.append( {
       "descripcion": row['descripcion'],
       "direccion":  respuestas["direccion"],
       "fot" : respuestas["fot"],
       "irregular": respuestas["irregular"],
       "medidas": respuestas["medidas"],
       "esquina": respuestas["esquina"],
       "barrio": respuestas["barrio"],
       "frentes": respuestas["frentes"],
       "pileta": respuestas["pileta"]
   } )


df= pd.DataFrame(data, index=None) 
df.set_index('descripcion', inplace=True)
df.to_csv("rbm/respuestas.csv", sep="|")   