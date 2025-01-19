import json
import pandas as pd
import os
import time
from tqdm.notebook import tqdm

DIR = "C:/Users/enduser/OneDrive - Politecnico di Milano/Ingegneria matematica/Tesi/ProveDiCodice/E3C-Corpus/data_collection/Italian/layer3"
print(time.time()) # t = 0
# Elenco dei file nella directory

data_files = os.listdir(DIR)

dfs = []
for filename in tqdm(os.listdir(DIR)):
    authors_string = False
    authors_dict = False
    f = os.path.join(DIR, filename)
    if os.path.isfile(f):
      with open(f, 'r', encoding='utf-8') as fp: # QUESTA è LA RIGA MODIFICATA, VEDI CODICE ORIGINALE VITTORIO PER CAPIRE, QUESTA MODIFICA è STATA NECESSARIA ALTRIMENTI MI DAVA ERRORE (SU COLLAB NO)
        d = json.load(fp)
      if d['authors'] == []:
        d['authors'] = ''
        i = [0]
      elif isinstance(d['authors'], str): #for some Spanish documents (es ES102568.json)
        authors_string = True
        i = [0]
      elif isinstance(d['authors'], dict): #for Basque
        authors_dict = True
        i = [0]
      else:
        i = list(range(len(d['authors'])))
      data = pd.DataFrame(d, index=i)
      if d['authors'] != '':
        try:
          data.authors = pd.DataFrame(data.authors.values.tolist())['author']
        except: #spanish has different format
          if authors_dict:
            data.authors = d['authors']['author']
          elif not authors_string:
            data.authors = pd.DataFrame(data.authors.values.tolist())[1]
        cols = list(data.columns) # columns are different in different languages
        cols.remove('authors')
        data = data.groupby(cols)['authors'].apply(','.join).reset_index()
      dfs.append(data) # append the data frame to the list
df = pd.concat(dfs, ignore_index=True, axis=0) # concatenate all the data frames in the list.

#print(time.time()) # t = 15

import nltk
#nltk.download('punkt')
def filter_words(sentence):
    return [word for word in nltk.word_tokenize(sentence) if word.isalnum()]

def tokenize(text):
    return [filter_words(sentence) for sentence in nltk.sent_tokenize(text.lower())]

documents = []
for i in range(len(df)):
    text = df.loc[i,'text']
    documents.append(text)

#documents sono tutti i documenti non tokenizzati ancora

sentences = []
for doc in documents:
    sentences.extend(tokenize(doc))

# sentences sono tutte le frasi di tutti i 10213 documenti

#print(time.time()) # t = 115 (quindi 100)

corpusAnnidato = []
for docu in documents:
    corpusAnnidato.append(tokenize(docu))

#print(time.time()) # t = 215 (quindi 100)


# corpusAnnidato è la lista di tutti i documenti, però ogni documento è una lista annidata di frasi che ora vado ad "appiattire", per avere che un documento sia una lista di parole, non una lista di lista di parole (ovvero lista di frasi)
# osservo che corpusAnnidato potrebbe tornare utile ad esempio per HypergraphGCN dove le frasi hanno una loro importanza!

corpus = [[parola for frase in documento for parola in frase] for documento in corpusAnnidato]

#corpus è una lista di 10213 documenti che sono rappresentati ciascuno da una lista di lunghezza variabile di parole tokenizzate.

def appiatta_lista(lista):
    lista_appiattita = []
    for elemento in lista:
        if isinstance(elemento, list):
            lista_appiattita.extend(appiatta_lista(elemento))
        else:
            lista_appiattita.append(elemento)
    return lista_appiattita

VocabolarioNonFiltrato = appiatta_lista(sentences)

# VocabolarioNonFiltrato è una lista semplice contenente tutte le parole trovate, con anche le parole inutili e tutte le ripetizioni di parole trovate

ListaParoleInutili = [
    "il", "la", "lo", "i", "gli", "le","un","uno","una", "del", "dello", "della", "dei", "degli", "delle",  # Articoli
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",  # Preposizioni
    "e", "o", "ma", "anche", "neanche", "nonché","e", "anche", "inoltre", "pure",
"anche", "né", "neanche",
"nemmeno", "neppure","o"," oppure"," ovvero",
"altrimenti","ma", "tuttavia", "però", "eppure",
"invece", "anzi", "nondimeno",
"bensì", "infatti", "difatti", "invero", "cioè","ossia","dunque", "perciò", "quindi",
"pertanto", "allora", "insomma",
"sicché"]# Congiunzioni

 # ListaParoleInutili è la lista di parole che non penso possano essere utili ai modelli futuri, quindi setacciamo il testo per 
 # poter togliere queste parole! (E' una lista provvisoria che deve essere estesa nel tempo in base anche all'utilità)



Vocabolario_filtrato = [parola for parola in VocabolarioNonFiltrato if parola not in ListaParoleInutili]

# Vocabolario_filtrato, contiene utte le parole utili, con anche la loro ripetizione

Vocabolario = set(Vocabolario_filtrato)

# Vocabolario è come intuibile il set (quindi c'è unicità!) delle parole utili che sono state trovate nel testo 

corpus_filtrato = [[parola for parola in doc if parola not in ListaParoleInutili] for doc in corpus]

# corpus filtrato è come corpus a cui abbiamo tolto le parole dalla lista delle parole inutili

print(time.time()) # t = 230 (ovvero 15)

# Analisi costo computazionale finale:
# Ci vogliono 230 secondi per eseguire questo codice, lo ho runnato più volte, non so se questo ha reso la velocità
# di esecuzione migliore.
# Come prevedibile, la tokenizzazione è il procedimento più lungo!
# SU colab con la gpu, più o meno il tempo è lo stesso (4 min in tutto) [fatto copia in colla di tutto il codice e messo sul notebook]


