Questa cartella contiene sia la versione originale di InfoGraph presa da github https://github.com/sunfanyunn/InfoGraph, che le mie modifiche.
Dovendo noi lavorare in un setting unsupervised, le mie modifiche sono solo dentro alla cartella unsupervised.

Il main per effettuare il clustering con setting transductive è il file mainClusteringèlayer1and3_transd. In poche parole questo file prende in input i grafi creati in DataPreprocessing sotto formato DataLoader, applica l'algoritmo di InfoGraph e alla fine di tutto salva gli embedding dei singoli grafi (ovvero documenti) che saranno poi valutati nei vari codici di valutazione della bontà degli embeddings presenti nella cartella "CodiciMiei".

All'interno di "CodiciMiei" oltre a script python per la valutazione degli embeddings, c'è la cartella FilePickles dove salvo e da dove prendo i dati di input e output per i vari codici.

Analogamente, "Immagini" contiene le immagini salvate per essere messe nella tesi scritta su latex. 