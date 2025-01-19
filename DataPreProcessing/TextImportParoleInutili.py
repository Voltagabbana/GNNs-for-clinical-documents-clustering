##############################################################################################################################

#Questo file è una modifica di TextImport che punta a migliorare la tokenizzazione nel senso che in TextImport abbiamo eliminato le parole
#inutili sulla base della nostra intuizione (congiunzioni articoli ecc), qua invece con from nltk.corpus import stopwords utilizziamo un risultato
#pregresso. 



import json
import pandas as pd
import os
import time

DIR = "C:/Users/enduser/OneDrive - Politecnico di Milano/Ingegneria matematica/Tesi/ProveDiCodice/E3C-Corpus/data_collection/Italian/layer3"
print(time.time()) # t = 0
# Elenco dei file nella directory

data_files = os.listdir(DIR)

dfs = []
for filename in data_files:
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

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import time
import pandas as pd

print(time.time())  # t = 15

nltk.download('stopwords')

def filter_words(sentence):
    return [word for word in nltk.word_tokenize(sentence) if word.isalnum()]

def tokenize(text):
    return [filter_words(sentence) for sentence in nltk.sent_tokenize(text.lower())]

# Aggiungi questa riga per inizializzare lo stemmer italiano
stemmer = SnowballStemmer("italian")

documents = []
for i in range(len(df)):
    text = df.loc[i, 'text']
    documents.append(text)

# Crea una lista di stopwords in italiano da nltk
stop_words = set(stopwords.words('italian'))

# Tokenizza i documenti, applica lo stemming e rimuovi le stopwords
corpus = []
for doc in documents:
    tokenized_doc = tokenize(doc)
    
    # Applica lo stemming
    stemmed_doc = [
        stemmer.stem(parola) for frase in tokenized_doc for parola in frase
        if parola not in stop_words
    ]
    
    corpus.append(stemmed_doc)

print(time.time())

import pickle

with open('corpusParoleInutiliStemming.pkl', 'wb') as file:
    pickle.dump(corpus, file)





'''


import nltk
import pandas as pd
from nltk.corpus import stopwords
import time

print(time.time())  # t = 15

nltk.download('stopwords')

def filter_words(sentence):
    return [word for word in nltk.word_tokenize(sentence) if word.isalnum()]

def tokenize(text):
    return [filter_words(sentence) for sentence in nltk.sent_tokenize(text.lower())]

documents = []
for i in range(len(df)):
    text = df.loc[i, 'text']
    documents.append(text)

# Crea una lista di stopwords in italiano da nltk
stop_words = set(stopwords.words('italian'))

# Tokenizza i documenti e rimuovi le stopwords
corpus = []
for doc in documents:
    tokenized_doc = tokenize(doc)
    cleaned_doc = [
        parola for frase in tokenized_doc for parola in frase if parola not in stop_words
    ]
    corpus.append(cleaned_doc)

print(time.time())

import pickle

with open('corpusParoleInutili.pkl', 'wb') as file:
    pickle.dump(corpus, file)
'''