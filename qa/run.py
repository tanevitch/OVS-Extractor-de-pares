
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper import procesar_medidas, procesar_direccion, procesar_fot, procesar_frentes, procesar_irregular, get_numeros
import pandas as pd
from transformers import pipeline
import spacy
PIPE = pipeline("question-answering", model="rvargas93/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es", handle_impossible_answer=True)

PREGUNTAS= [
    "¿Cuál es la dirección del lote?",
    "¿Cuál es el valor del FOT?",
    "¿El terreno es de forma irregular?",
    "¿Cuáles son las dimensiones del lote?",
    "¿El lote está en una esquina?",
    "¿En qué barrio privado está ubicado?",
    "¿Cuántos frentes tiene el inmueble?",
    "¿El inmueble tiene piscina?"
]
NLP = spacy.load("es_core_news_lg")

input = pd.read_csv('ground_truth_100_sin_inferencias.csv', sep = '|')
input = input.fillna("")

data= []
for index, row in input.iterrows():
    respuestas= PIPE(question=PREGUNTAS, context=row["descripcion"], handle_impossible_answer=True)
    respuestas = [[' '.join([token.text for token in NLP(rta["answer"]) if not token.is_punct]).strip()] for rta in respuestas]

    respuestas[1]= procesar_fot(respuestas[1]) if respuestas[1] else ""
    respuestas[2]= procesar_irregular(respuestas[2]) if respuestas[2] else ""
    respuestas[3]= procesar_medidas(respuestas[3]) if respuestas[3] else ""
    respuestas[4]= True if "esquina" in respuestas[4][0] else "" 
    respuestas[6]= procesar_frentes(respuestas[6]) if respuestas[6] else ""
    respuestas[7]= True if any(map(lambda subs: subs.lower() in respuestas[7][0].lower(), ["piscina", "pileta"])) else ""

    data.append( {
       "descripcion": row['descripcion'],
       "direccion":  respuestas[0][0],
       "fot" : respuestas[1],
       "irregular": respuestas[2],
       "medidas": respuestas[3],
       "esquina": respuestas[4],
       "barrio": respuestas[5][0],
       "frentes": respuestas[6],
       "pileta": respuestas[7]
   } )

df= pd.DataFrame(data, index=None) 
df.set_index('descripcion', inplace=True)
df.to_csv("qa/respuestas_rvargas93.csv", sep="|")   