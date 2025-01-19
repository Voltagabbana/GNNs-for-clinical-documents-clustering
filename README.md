# Graph Neural Networks for clinical documents clustering
Master thesis for Mathematical Engineering - Statistical Learning. 
This thesis work has been carried on by me with the help of dott. Vittorio Torri. The code structure is not the best for reproducibility but serves to give an idea of the work done in the period from September 2023 and June 2024.
For a short explanation of the thesis work check out the "Executive Summary" file. For a deeper understanding check the "Tesi" file

The code is organized into 3 main folders:
- DataPreProcessing
- WordEmbeddings
- InfoGraph


### DataPreProcessing
In this folder the clinical documents of E3C (https://github.com/hltfbk/E3C-Corpus) have been preprocessed, performing in order, tokenization, removal of stop-words and stemming. Then the documents have been represented in a graph structure.

### WordEmbeddings
The preprocessed words are translated into a mathematical representation using Word2Vec (https://arxiv.org/abs/1301.3781)

### InfoGraph
InfoGraph is an unsupervised approach based on Graph Neural Networks for representing a graph in a mathematical form (https://github.com/sunfanyunn/InfoGraph). The code has been modified to take account of the needs of the task.

