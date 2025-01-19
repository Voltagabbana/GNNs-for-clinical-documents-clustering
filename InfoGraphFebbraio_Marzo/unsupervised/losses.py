import torch
import torch.nn as nn
import torch.nn.functional as F
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map. #\nodi
        g: Global features. #\grafi
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device) #\di default è zero
    neg_mask = torch.zeros((num_nodes, num_graphs)).to(device) #\di default è zero
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1. #\ quando siamo nel caso corretto assegno 1 (rendo positivo)
        neg_mask[nodeidx][graphidx] = 0. #\ quando siamo nel caso scorretto assegno 0 (rendo zero)

    res = torch.mm(l_enc, g_enc.t()) #\ matrix multiplication

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

'''
Inizia inizializzando maschere (pos_mask e neg_mask) per separare gli esempi positivi da quelli negativi.
Usa un doppio loop per creare queste maschere basate su batch. Ogni nodo è etichettato come positivo per il proprio grafo e negativo per gli altri grafi nello stesso batch.
Calcola il prodotto interno tra le feature locali e globali (torch.mm(l_enc, g_enc.t())), ottenendo una matrice di similarità tra i nodi locali e globali.
Calcola le aspettative positive (E_pos) e negative (E_neg) utilizzando la funzione get_positive_expectation e get_negative_expectation rispettivamente. Queste funzioni sembrano calcolare aspettative basate su una divergenza specifica (misurata da measure).
La perdita finale è la differenza tra le aspettative negative e positive normalizzate rispetto al numero di nodi.
'''

def adj_loss_(l_enc, g_enc, edge_index, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    adj = torch.zeros((num_nodes, num_nodes)).to(device)
    mask = torch.eye(num_nodes).to(device)
    for node1, node2 in zip(edge_index[0], edge_index[1]):
        adj[node1.item()][node2.item()] = 1.
        adj[node2.item()][node1.item()] = 1.

    res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
    res = (1-mask) * res
    # print(res.shape, adj.shape)
    # input()

    loss = nn.BCELoss()(res, adj)
    return loss

'''Inizializza una matrice di adiacenza (adj) con zeri, dove ogni elemento rappresenta la presenza di un arco tra due nodi.
Crea una maschera (mask) con uno sulla diagonale e zeri altrove. Questa maschera garantisce che i self-loop non siano considerati nel calcolo della perdita.
Itera attraverso gli indici degli archi (edge_index) e imposta gli elementi corrispondenti nella matrice di adiacenza a 1, indicando la presenza di un arco.
Applica la funzione sigmoide al prodotto scalare delle caratteristiche locali (torch.mm(l_enc, l_enc.t())). Questa operazione è probabilmente utilizzata per calcolare la matrice di adiacenza prevista.
Moltiplica la matrice risultante per l'inverso della maschera per annullare gli elementi sulla diagonale (self-loop).
Calcola la perdita di entropia binaria tra la matrice di adiacenza prevista e quella effettiva utilizzando nn.BCELoss().
Questa perdita è comunemente utilizzata in compiti relativi ai grafi per misurare la dissimilarità tra le matrici di adiacenza previste ed effettive. L'obiettivo è generalmente quello di addestrare il modello a predire con precisione gli archi nel grafo.'''