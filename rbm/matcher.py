
import copy
import spacy
from spacy.matcher import Matcher as MatcherSpacy
from spacy.matcher import DependencyMatcher as DependencyMatcherSpacy

from helper import procesar_barrio, procesar_direccion, procesar_fot, procesar_frentes, procesar_irregular, procesar_medidas

NLP = spacy.load('es_core_news_lg')

class Matcher():
    
    def __init__(self) -> None:
        if Matcher.matcher is None:
            Matcher.initialize_matcher()

    matcher= None
    dependencyMatcher= None

    @staticmethod
    def initialize_matcher():
        Matcher.matcher = MatcherSpacy(NLP.vocab)
        Matcher.matcher.add('medidas', [[
            {'LIKE_NUM': True}, 
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
            {'LOWER': 'x'}, 
            {'LIKE_NUM': True},
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
            {'LOWER': 'x', 'OP': '?'}, 
            {'LIKE_NUM': True, 'OP': '?'},
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
            {'LOWER': 'x', 'OP': '?'}, 
            {'LIKE_NUM': True, 'OP': '?'},
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
        ],
        [
            {'LIKE_NUM': True}, 
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
            {'LOWER': 'x'}, 
            {'LIKE_NUM': True},
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
            {'OP': '?'}, 
            {'LOWER': 'martillo'},
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
            {'LOWER': 'x', 'OP': '?'}, 
            {'LIKE_NUM': True, 'OP': '?'},
            {'LOWER': {"IN": ["mts","m","metros"]}, "OP":"?"},
        ],
        [   
            {'LIKE_NUM': True},
            {'LOWER': {'IN': ['m', 'mts', 'metros']}, "OP": "?"},
            {"LOWER": "de", "OP": "?"},
            {'LOWER': 'frente'},
            {'LOWER': {'IN': ['por', 'x', 'y']}, "OP": "?"},
            {'LIKE_NUM': True},
            {'LOWER': {'IN': ['m', 'mts', 'metros']}, "OP": "?"},
            {"LOWER": "de", "OP": "?"},
            {'LOWER': 'fondo'}
        ]])
        Matcher.matcher.add('pileta', [[
            {'LEMMA': {'IN': ['piscina', 'pileta']}}, 
        ]])
        Matcher.matcher.add('esquina', [[
            {'LOWER': 'esquina'}, 
        ]])
        Matcher.matcher.add('irregular', [[
            {'LEMMA': 'irregular'}
        ],
        [
        {'LOWER': {'IN': ['lote', 'forma']}},
        {'POS': 'ADJ'}
        ]])
        Matcher.matcher.add('fot', [[
            {'LOWER': {'IN':['fot', 'f.o.t']}},
            {'LOWER':  {'IN':['res', 'residencial', 'comercial', 'com', 'industrial']}, "OP": "?"},
            {"IS_PUNCT": True, "OP":"?"},
            {"LIKE_NUM": True}
        ]])
        Matcher.matcher.add('barrio', [[
            {"LOWER": {'IN': ['estancia', 'barrio', 'country', 'club']}},
            {'POS':"PROPN", "OP": "+"}
        ]])
        Matcher.matcher.add('dir_nro', [ [
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},   
            {'TEXT': '.', 'OP':'?'},  
            {'POS': 'PROPN', 'OP': '+'},  
            {'LOWER': 'al', "OP": "?"}, 
            {'LIKE_NUM': True}    
        ],
        [
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},   
            {'TEXT': '.', 'OP':'?'}, 
            {'POS': 'PROPN', 'OP': '+'},  
            {'LOWER': 'n'}, 
            {'TEXT': '°'}, 
            {'LIKE_NUM': True}    
        ]])
        Matcher.matcher.add('dir_interseccion', [[
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},   
            {'TEXT': '.', 'OP':'?'}, 
            {'POS': {'IN': ['PROPN', 'NUM']}, 'OP': '+'},  
            {'LOWER': {"IN": ["y", "esquina", "esq.", "e"]}},  
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},   
            {'TEXT': '.', 'OP':'?'},      
            {'POS': {'IN': ['PROPN', 'NUM']}, 'OP': '+'},     
        ]])
        Matcher.matcher.add('dir_entre', [[
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},  
            {'TEXT': '.', 'OP':'?'}, 
            {'POS': {'IN': ['PROPN', 'NUM']}, 'OP': '+'},   
            {'LOWER': {'IN': ['e/', 'entre', 'e']}},
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},   
            {'TEXT': '.', 'OP':'?'}, 
            {'POS': {'IN': ['PROPN', 'NUM']}, 'OP': '+'},  
            {'LOWER': 'y'},    
            {'LOWER': {'IN': ['calle', 'avenida', 'av', 'diagonal', 'diag']}, 'OP':'?'},   
            {'TEXT': '.', 'OP':'?'}, 
            {'POS': {'IN': ['PROPN', 'NUM']}, 'OP': '+'},  
        ]])
        Matcher.matcher.add('dir_lote', [[
            {'LOWER': 'lote'},
            {'POS': {'IN': ['NUM', 'PROPN']}}
        ]])

        Matcher.dependencyMatcher = DependencyMatcherSpacy(NLP.vocab)
        Matcher.dependencyMatcher.add('frentes', patterns=[
            [
                {'RIGHT_ID': 'frentes', 'RIGHT_ATTRS': {'LOWER': {'IN':['frente', 'frentes']}}}, 
                {'LEFT_ID': 'frentes', 'REL_OP': '>', 'RIGHT_ID': 'num', 'RIGHT_ATTRS': {'DEP': {"IN": ['nummod', 'amod']}}} 
            ],
            [
                {'RIGHT_ID': 'frentes', 'RIGHT_ATTRS': {'LOWER': 'salida'}},
                {'LEFT_ID': 'frentes', 'REL_OP': '>', 'RIGHT_ID': 'calles', 'RIGHT_ATTRS': {'DEP': 'obl'}}, 
                {'LEFT_ID': 'calles', 'REL_OP': '>', 'RIGHT_ID': 'num', 'RIGHT_ATTRS': {'DEP': 'nummod'}} 
            ] 
        ])
        Matcher.dependencyMatcher.add('fot', patterns=[
        [
            {'RIGHT_ID': 'fot', 'RIGHT_ATTRS': {'LOWER': {'IN':['fot', 'f.o.t']}}},
            {'LEFT_ID': 'fot', 'REL_OP': '>', 'RIGHT_ID': 'num', 'RIGHT_ATTRS': {'DEP': 'nummod'}} 
        ]
    ])

    def __get_matches(self, text, prev_result):
        doc= NLP(text)

        matches = Matcher.matcher(doc)
        for match_id, start, end in matches:
            matched_span = doc[start:end]
            prev_result[NLP.vocab.strings[match_id]].append(matched_span.text)

    def __get_dep_matches(self, text, prev_result): 
        doc= NLP(text)
        matches_dep = Matcher.dependencyMatcher(doc)
        for match_id, token_ids in matches_dep:
            palabra= []
            for token_id in sorted(token_ids):
                token = doc[token_id]
                palabra.append(token.text)
            prev_result[NLP.vocab.strings[match_id]].append(' '.join(palabra))

    def __merge(self, dic1, dic2):
        for clave, valores in dic1.items():
            if clave in dic2:
                dic2[clave].extend(valores)
            else:
                dic2[clave] = valores
        
        
        return dic2
    

    def obtener_mejor_resultado(self, predichos):
        return {
            'direccion': procesar_direccion(predichos),
            'fot': procesar_fot(predichos["fot"]) if predichos["fot"] else "",
            'irregular': procesar_irregular(predichos["irregular"]) if len(predichos["irregular"])>0 else "",
            'medidas': procesar_medidas(predichos["medidas"]) if predichos["medidas"] else "",
            'esquina': True if len(predichos["esquina"])>0 else "",
            'barrio': procesar_barrio(predichos["barrio"]) if predichos["barrio"] else "",
            'frentes': procesar_frentes(predichos["frentes"]) if predichos["frentes"] else "",
            'pileta': True if len(predichos["pileta"])>0 else "",
        }
            
    def get_pairs(self, text: str):
        prev_result = {
            'medidas': [],
            'dir_nro': [],
            'dir_interseccion': [],
            'dir_entre': [],
            'dir_lote': [],
            'fot': [],
            'irregular': [],
            'pileta': [],
            'barrio': [],
            'esquina': [],
            'frentes': []
        }
        self.__get_matches(text, prev_result)
        self.__get_dep_matches(text, prev_result)        

        a= self.obtener_mejor_resultado(prev_result)
        return a
        
