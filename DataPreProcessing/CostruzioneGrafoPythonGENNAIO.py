#### NOTA BENE ####

## questo codice non è altro che il copia-incolla di"costruzioneGrafo.ipynb", devo fare così per poter uploaddare 
## futuro questo modulo (posso importare.py ma non .ipynb). (In reltà il codice qua è molto più corto perchè dovevo 
# importare solo alcune cose!!)
#import gensim

### Nota bene 2 (modifiche di fine gennaio): #########

# sto cercando ora di modificare i grafi per non essere solon grafi di linea ma anzi essere del tipo di text level
# gcn dove la matrice di adiacenza è sulla base delle parole adiacenti con una finestra "p"

# per valutare la bonta della produzione vado sul file .ipynb che analizza nel dettaglio i dataset di grafo creati
# cosicche sfrutto i file pickle e non devo rirunnare tutto il codice!

import time
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import gensim
import tqdm


#with open('corpus_filtrato15_11.pkl', 'rb') as file:
    # Carica i dati dal file
 #   corpus_filtrato = pickle.load(file) # corpus filtrato è il corpus senza le parole inutili

with open('corpusParoleInutiliStemming.pkl', 'rb') as file:
    # Carica i dati dal file
    corpus_filtrato = pickle.load(file) # corpus filtrato è il corpus senza le parole inutili
#with open("my_vecs_preProcessingSenzaParoleInutili.p", "rb") as file:
   #word2Vec_model = pickle.load(file)
with open("my_vecs_GennaioStemming.p", "rb") as file:
    word2Vec_model = pickle.load(file)
word_names = word2Vec_model.wv.index_to_key # word names è la stringa di parola corrispondente all'embedding numerico
# (Traduzione Embedding -> stringa)
word_vectors = [word2Vec_model.wv[word] for word in word_names]
# word_vectors è l'embedding relativo alle parole di word_names
corpus_filtrato_embedding = []
print('ciao')
for doc in corpus_filtrato:
    doc_embedding = []
    for parola in doc:
        doc_embedding.append(word2Vec_model.wv[parola])
    corpus_filtrato_embedding.append(doc_embedding)

datasetGrafi = []

for Doc in corpus_filtrato_embedding:
    Doc = np.array(Doc)
    A = np.eye(len(Doc), dtype=int)  # Creazione della matrice di adiacenza con diagonale piena di 1
    A[:-1, 1:] += np.eye(len(Doc) - 1, dtype=int)  # Assegnazione dei 1 agli elementi sopra la diagonale
    A[1:, :-1] += np.eye(len(Doc) - 1, dtype=int)  # Assegnazione dei 1 agli elementi sotto la diagonale

    edge_index = torch.tensor(np.column_stack(np.where(A == 1)), dtype=torch.long)
    x = torch.tensor(Doc, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    datasetGrafi.append(data)
print('ciao')

data_load = DataLoader(datasetGrafi, batch_size=128)

#ora voglio cambiare questo codice nella seguente maniera: i collegamenti delle parole sono solo fra la parola antecedente e successiva 
#alla parola in analisi, io voglio invece che si faccia per le "p" parole successive e precedenti. Inoltre una notifica importante è che 
#se la stessa parola ricapita, non si deve aggiungere un elemento alla matrice di adiacenza ma riutilizzando l'indice della parola già
#trovata, si vada ad aggiornare i collegamenti

print('ciao')

datasetGrafiWindow = []  # Assicurati di avere questa lista definita

p = 2  # Numero di parole successive e precedenti

for doc in corpus_filtrato:
    doc_array = np.array(doc)
    
    # Creazione di un dizionario per tracciare gli indici delle parole e i collegamenti
    word_index = {}
    edges = []

    # Creazione dei collegamenti tra parole successive e precedenti
    for i, word in enumerate(doc_array):
        for j in range(-p, p + 1):
            if j != 0 and 0 <= i + j < len(doc_array):
                neighbor_word = doc_array[i + j]
                if neighbor_word in word_index:
                    neighbor_index = word_index[neighbor_word]
                else:
                    neighbor_index = len(word_index)
                    word_index[neighbor_word] = neighbor_index

                edges.append((i, neighbor_index))

    # Converti gli edge in tensor edge_index
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Crea il tensor x con le features delle parole
    # Converti le stringhe in indici numerici
    x = torch.tensor([word_index[word] for word in doc_array], dtype=torch.float)

    # Crea l'oggetto Data e aggiungilo al datasetGrafiWindow
    data = Data(x=x, edge_index=edge_index)
    datasetGrafiWindow.append(data)


data_loadWindow = DataLoader(datasetGrafiWindow, batch_size=128)

print("OK running finito")
print(data_load)
print(data_loadWindow)

print(len(datasetGrafi))
print(len(datasetGrafiWindow))
print(len(datasetGrafi[0]))
print(len(datasetGrafiWindow[0]))
print(len(datasetGrafi[1]))
print(len(datasetGrafiWindow[1]))



import pickle

# Save datasetGrafi to a pickle file
with open('datasetGrafi.pkl', 'wb') as file:
    pickle.dump(datasetGrafi, file)

# Save data_load to a pickle file
with open('data_load.pkl', 'wb') as file:
    pickle.dump(data_load, file)

# Save datasetGrafi to a pickle file
with open('datasetGrafiWindow.pkl', 'wb') as file:
    pickle.dump(datasetGrafiWindow, file)

# Save data_load to a pickle file
with open('data_loadWindow.pkl', 'wb') as file:
    pickle.dump(data_loadWindow, file)