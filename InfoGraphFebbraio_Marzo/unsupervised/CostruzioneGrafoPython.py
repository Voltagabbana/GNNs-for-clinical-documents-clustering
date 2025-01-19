#### NOTA BENE ####

## questo codice non è altro che il copia-incolla di"costruzioneGrafo.ipynb", devo fare così per poter uploaddare 
## futuro questo modulo (posso importare.py ma non .ipynb). (In reltà il codice qua è molto più corto perchè dovevo 
# importare solo alcune cose!!)
#import gensim
import time
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import gensim


with open('corpus_filtrato15_11.pkl', 'rb') as file:
    # Carica i dati dal file
    corpus_filtrato = pickle.load(file) # corpus filtrato è il corpus senza le parole inutili
with open("my_vecs_preProcessingSenzaParoleInutili.p", "rb") as file:
    word2Vec_model = pickle.load(file)
word_names = word2Vec_model.wv.index_to_key # word names è la stringa di parola corrispondente all'embedding numerico
# (Traduzione Embedding -> stringa)
word_vectors = [word2Vec_model.wv[word] for word in word_names]
# word_vectors è l'embedding relativo alle parole di word_names
corpus_filtrato_embedding = []
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

data_load = DataLoader(datasetGrafi, batch_size=128)

print("OK running finito")
print(data_load)
print(len(datasetGrafi))


