#################################### NOTA BENE ##########################################

# QUESTO CODICE PRODUCE EMBEDDING IN SETTING TRANSDUCTIVE CON LAYER 1 E LAYER 3!

# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from argumentsClustering import arg_parse
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json
import json
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(InfoGraph, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(10, hidden_dim, num_gc_layers) # metto 10 al posto di "dataset_num_features"

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch, num_graphs):
    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

if __name__ == '__main__':
    args = arg_parse()
    #accuracies = {'logreg':[], 'svc':[]} #'linearsvc':[], 'randomforest':[]
    epochs = args.epochs
    log_interval = 1
    batch_size = 128
    lr = args.lr
    hidden_dim = args.hidden_dim
    num_gc_layers = args.num_gc_layers

    import time
    import pickle
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    import gensim

    #with open('datasetGrafi.pkl', 'rb') as file:
    #    loaded_datasetGrafi = pickle.load(file)
    # data_load = DataLoader(loaded_datasetGrafi, batch_size=128)

    # Il codice commentato sopra è perchè è analogo a quello che facciamo appena qua sotto!

    #with open('CodiciMiei\FilePickles\data_load_NO_ripetizioni_layer1and_layer3.pkl', 'rb') as file: #data_load contiene i dati a cui è già stato applicato dataLoader
     #   loaded_data_load = pickle.load(file)
    # stringa_AIFA = ''
    
    # LOADDO i dati di layer 1 e 3 rimuovendo i foglietti illustrativi:
    with open('CodiciMiei\FilePickles\data_load_NO_ripetizioniItalianJournalofMedicinelayer3.pkl', 'rb') as file: #data_load contiene i dati a cui è già stato applicato dataLoader
        loaded_data_load = pickle.load(file)
    stringa_AIFA = '_senza_AIFA'


    
    dataloader = loaded_data_load
    print("Dataloader .pkl caricato")
    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(10))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.eval()
    emb = model.encoder.get_embeddings(dataloader) #\ get embdeddings è creata su gin.py, a noi serve solo emb, non y!!
    #print(len(emb[1])) # osservo che emb sono 188 per mutag quindi ogni grafo è un array di numeri dentro ad "emb",
    # la dimensione di ciascun vettore è 96 (3 gnn layers che producono vettori dim 32 che concatenandosi 3 volte arrivano a 96)
    #\ non servono accuracies per clustering! -> commento sotto:
    '''
    print('===== Before training =====')
    res = evaluate_embedding(emb, y)
    accuracies['logreg'].append(res[0])
    accuracies['svc'].append(res[1])
    #\accuracies['linearsvc'].append(res[2])
    #\accuracies['randomforest'].append(res[3])
    

    '''
    labels = None
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader: #\ per ogni data in dataloader fa unsupervised learning con infograph
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))
        '''
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)
            res = evaluate_embedding(emb, y)
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            #\accuracies['linearsvc'].append(res[2])
            #\accuracies['randomforest'].append(res[3])
            print(accuracies)'''

    #\ faccio clustering k-means invece che accuracies!
        
        if epoch % log_interval == 0:  #\ se si entra nell'if fa predizione unsupervised (clustering) dell'embedding 
            #\ avendo log_interval == 1, questo if in realtà è inutile!
            model.eval()
            emb = model.encoder.get_embeddings(dataloader)
            emb_array = np.array(emb)
            
# NB'MeanPool' è perchè ho cambiato la readout function in gin.py da somma a media!

#with open('CodiciMiei\FilePickles\EmbeddingLayer1and3_InfoGraphTRANSDUCTIVE'+'hidden_dim'+str(hidden_dim)+'num_gc_layers'+str(num_gc_layers)+'lr'+str(lr)+str(stringa_AIFA)+'MeanPool'+'.pkl', 'wb') as file:
 #   pickle.dump(emb_array, file)
    #\ mi salvo gli embedding ottenuti per poterci fare le analisi di clustering ecc in un altro file (per semplicità e per non dover
    #\ rirunnare sempre tutto)


with open('CodiciMiei\FilePickles\Embedding3_InfoGraphTRANSDUCTIVE'+'hidden_dim'+str(hidden_dim)+'num_gc_layers'+str(num_gc_layers)+'lr'+str(lr)+'ItalianJournalofMedicine'+'.pkl', 'wb') as file:
    pickle.dump(emb_array, file)
    #\ mi salvo gli embedding ottenuti per poterci fare le analisi di clustering ecc in un altro file (per semplicità e per non dover
    #\ rirunnare sempre tutto)

    
#with open('unsupervised.log', 'a+') as f:
 #   s = json.dumps(loss)
 #  f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))


# Esempio er runnare da terminale/command line (vedi argument, parser):
    # python mainClustering.py --epochs 6 --lr 0.01 --num-gc-layers 3
