import spacy
import re 

NLP = spacy.load("es_core_news_lg")
dos= re.compile(r'\b(dos|doble|2|segundo)\b', re.IGNORECASE)
tres= re.compile(r'\b(tres|triple|3|tercer)\b', re.IGNORECASE)
def clear_altura_entre(predichos: list):
    if predichos["dir_nro"] and predichos["dir_entre"]:
        dir_altura=max(predichos["dir_nro"], key=len)
        dir_entre=max(predichos["dir_entre"], key=len)
        if dir_entre.startswith(dir_altura):
            predichos["dir_entre"]= [dir_entre]
        else:
            numero = dir_altura.split()[-1]
            dir_entre= dir_entre.replace(numero, "").strip()
            predichos["dir_entre"]= [ dir_altura+ dir_entre]
            predichos["dir_nro"]= []
    return predichos

def clear_inter_entre(result: list):
    for interseccion in result.get('dir_interseccion', []):
        for entre in result.get('dir_entre', []):
            if interseccion in entre and interseccion in result["dir_interseccion"]:
                result['dir_interseccion'].remove(interseccion)
    return result

def procesar_direccion(predichos: list):
    predichos= clear_inter_entre(predichos)
    # predichos = clear_altura_entre(predichos)
    matches_direccion_todos=predichos["dir_entre"]+predichos["dir_interseccion"]+predichos["dir_nro"]+predichos["dir_lote"]
    if matches_direccion_todos == []:
        return ""
    mejor_match= max(matches_direccion_todos, key=len)
    
    return re.sub(r'^\. ', '',mejor_match)

def procesar_fot(predichos: list):
    predichos= list(set(predichos))
    numeros = get_numeros(" ".join(predichos))
    if len(get_numeros((" ".join(predichos)))) == 1:
        unidad = re.search(r'\b(m2|mts2|mt2)\b', " ".join(predichos))
        if unidad:
            return " ".join(set(numeros))+" "+unidad.group()
        
        return " ".join(set(numeros))
    else: 
        if len(predichos) == 2:
            result = predichos[0]+". "+predichos[1]
            result = result.replace("Res.", "residencial:")
            result = result.replace("Com.", "comercial:")
            result = result.replace("Fot", "FOT")
            result = result.replace("fot", "FOT")
            return result
        else:
            result= "".join(predichos).rstrip(".")
            result = result.replace("Fot", "FOT")
            result = result.replace("fot", "FOT")
            return result


def procesar_irregular(predichos: list):
    for predicho in predichos:
        if any(map(lambda subs: subs.lower() in predicho.lower(), ["irregular", "triangular","martillo","trapecio"])):
            return True
    return ""
    
def get_numeros(cadena: str):
    return re.findall(r'\b\d+(?:[.,]\d+)?\b', cadena)
     
def contiene_dos(texto):
    return bool(dos.findall(texto))

def contiene_tres(texto):
    return bool(tres.findall(texto))

def procesar_frentes(frentes_predichos):
    frentes_en_numeros= []
    for match in frentes_predichos:
        contiene_2= contiene_dos(match.lower())
        if contiene_2:
            frentes_en_numeros.append(2)
        else: 
            contiene_3= contiene_tres(match.lower())
            if contiene_3:
                frentes_en_numeros.append(3)
    

    return max(frentes_en_numeros) if len(frentes_en_numeros) != 0 else ""



def procesar_medidas(predichos: list):
    mejor_match = max(predichos, key=len)
    if "martillo" in mejor_match:
        return mejor_match.replace(" mts", "")

    medidas = ""
    for numero in list(map(str, get_numeros(mejor_match))):
        medidas+= numero+" x "

    return medidas.rstrip(" x")

def procesar_barrio(predichos: list):
    return max(predichos, key=len)
    # return re.compile(re.escape("Barrio"), re.IGNORECASE).sub("", mejor_match).strip()