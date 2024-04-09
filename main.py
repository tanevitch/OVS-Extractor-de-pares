import spacy

import pandas as pd
NLP = spacy.load("es_core_news_lg")

gt = pd.read_csv('ground_truth_100.csv', sep = '|')
rtas = pd.read_csv('rbm/respuestas.csv', sep = '|')

gt = gt.fillna("")
rtas = rtas.fillna("")

metricas = {
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

columnas_comunes = set(gt.columns) & set(rtas.columns)

# Itera simultáneamente sobre las filas de ambos archivos
for rta, esperada in zip(rtas.itertuples(index=False), gt.itertuples(index=False)):
    for metrica in metricas:
        metrica_valor_rta = getattr(rta, metrica)
        metrica_valor_esperada = getattr(esperada, metrica)
        if (metrica_valor_rta == "" and metrica_valor_esperada==""):
            metricas[metrica]["tn"] += 1
        else:      
            if (NLP(str(metrica_valor_rta).lower()).similarity(NLP(str(metrica_valor_esperada).lower()))) >0.9:
                metricas[metrica]["tp"] += 1
            else:
                metricas[metrica]["error"].append({
                    "contexto": esperada.descripcion,
                    "respuesta_predicha": metrica_valor_rta,
                    "respuesta_esperada": metrica_valor_esperada
                })
                if metrica_valor_rta == "" and metrica_valor_esperada != "":
                    metricas[metrica]["fn"] += 1
                elif (metrica_valor_esperada == "" and metrica_valor_rta != "") or (metrica_valor_esperada != metrica_valor_rta):
                    metricas[metrica]["fp"] += 1
                


for metrica, valores in metricas.items():
    tp = valores["tp"]
    fp = valores["fp"]
    fn = valores["fn"]

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1_score = 0.0

    metricas[metrica]["p"] = precision
    metricas[metrica]["r"] = recall
    metricas[metrica]["f1"] = f1_score

import json
with open('resultados.json', 'w', encoding="utf8") as fp:
    json.dump(metricas, fp, ensure_ascii=False)