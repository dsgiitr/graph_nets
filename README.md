<h1 align="center"> Graph Representation Learning </h1>

This repo is a supplement to our blog series *Explained: Graph Representation Learning*. The following major papers and corresponding blogs have been covered as part of the series and we look to add blogs on a few other significant works in the field.

<h2 align="center"> 1. Understanding DeepWalk </h2>
<img align="right" width="500x" height="120x" src="https://miro.medium.com/max/4005/1*j-P55wBp5PP9oqrxDxdDpw.png">

Unsupervised online learning approach, inspired from word2vec in NLP, but, here the goal is to generate node embeddings.
- [DeepWalk Blog](https://dsgiitr.com/blogs/deepwalk)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/DeepWalk/DeepWalk_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/DeepWalk/DeepWalk.py)
- [Paper -> DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)


<h2 align="center"> 2. A Review : Graph Convolutional Networks (GCN) </h2>
<img align="right" width="500x" src="/GCN/img/gcn_architecture.png">

GCNs draw on the idea of Convolution Neural Networks re-defining them for the non-euclidean data domain.
- [GCN Blog](https://dsgiitr.com/blogs/gcn)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GCN/GCN_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/GCN/GCN.py)
- [Paper -> Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)






<h2 align="center"> 3. Graph SAGE(SAmple and aggreGatE) </h2>
<img align="right" width="500x" src="/GraphSAGE/img/GraphSAGE_cover.jpg">

Previous approaches are transductive and don't naturally generalize to unseen nodes. GraphSAGE is an inductive framework leveraging node feature information to efficiently generate node embeddings.
- [GraphSAGE Blog](https://dsgiitr.com/blogs/graphsage)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GraphSAGE/GraphSAGE_Code%2BBlog.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/GraphSAGE/GraphSAGE.py)
- [Paper -> Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)


<h2 align="center"> 4. ChebNet: CNN on Graphs with Fast Localized Spectral Filtering </h2>
<img align="right" width="600x" src="https://storage.googleapis.com/groundai-web-prod/media/users/user_3036/project_14426/images/x1.png">

ChebNet is a formulation of CNNs in the context of spectral graph theory.
- [ChebNet Blog](https://dsgiitr.com/blogs/chebnet/)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/ChebNet/Chebnet_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/ChebNet/coarsening.py)
- [Paper -> Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)

<br/>

<h2 align="center"> 5. Understanding Graph Attention Networks </h2>
<img align="right" width="500x" src="/GAT/img/GAT_Cover.jpg">

GAT is able to attend over their neighborhoods’ features, implicitly specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation or depending on knowing the graph structure upfront.
- [GAT Blog](https://dsgiitr.com/blogs/gat)
- [Jupyter Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GAT/GAT_Blog%2BCode.ipynb)
- [Code](https://github.com/dsgiitr/graph_nets/blob/master/GAT/GAT_PyG.py)
- [Paper -> Graph Attention Networks](https://arxiv.org/abs/1710.10903)
