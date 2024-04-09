
from helper import procesar_medidas, procesar_direccion, procesar_fot, procesar_frentes, procesar_irregular, get_numeros
import pandas as pd
from transformers import pipeline

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


data= []
for index, row in input.iterrows():
    respuestas= PIPE(question=PREGUNTAS, context=row["descripcion"], handle_impossible_answer=True)
    respuestas = [[rta["answer"]] for rta in respuestas]

    respuestas[1]= procesar_fot(respuestas[1]) if respuestas[1] else ""
    respuestas[2]= procesar_irregular(respuestas[2]) if respuestas[2] else ""
    respuestas[3]= procesar_medidas(respuestas[3]) if respuestas[3] else ""
    respuestas[4]= True if "esquina" in respuestas[4][0] else "" 
    respuestas[6]= procesar_frentes(respuestas[6]) if respuestas[6] else ""
    respuestas[7]= True if any(map(lambda subs: subs.lower() in respuestas[7][0].lower()), ["piscina", "pileta"]) else ""

    data.append( {
       "descripcion": row['descripcion'],
       "direccion":  respuestas[0],
       "fot" : respuestas[1],
       "irregular": respuestas[2],
       "medidas": respuestas[3],
       "esquina": respuestas[4],
       "barrio": respuestas[5],
       "frentes": respuestas[6],
       "pileta": respuestas[7]
   } )

df= pd.DataFrame(data, index=None) 
df.set_index('descripcion', inplace=True)
df.to_csv("respuestas.csv", sep="|")   