---
title: "Understanding Graph Attention Networks (GAT)"
date: 2020-01-01T23:40:49+00:00
description : "Machine Learning / Graph Representation Learning"
type: post
image: images/blogs/GAT/GAT_Cover.jpg
author: Anirudh Dagar
tags: ["Graph Representation Learning"]
---


<!-- # Understanding Graph Attention Networks (GAT) -->
<h1><center>Understanding Graph Attention Networks (GAT)</center></h1>

<!-- ![GAT Cover](GAT_Cover.jpg) -->
<!-- <img src="images/blogs/GAT/GAT_Cover.jpg" width=700x/> -->

This is 4th in the series of blogs <font color="green"><b>Explained: Graph Representation Learning</b></font>. Let's dive right in, assuming you have read the first three. GAT (Graph Attention Network), is a novel neural network architecture that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods’ features, the method enables (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, GAT addresses several key challenges of spectral-based graph neural networks simultaneously, and make the model readily applicable to inductive as well as transductive problems.

Analyzing and Visualizing the learned attentional weights also lead to a more interpretable model in terms of importance of neighbors.

But before getting into the meat of this method, I want you to be familiar and thorough with the Attention Mechanism, because we'll be building GATs on the concept of <b>Self Attention</b> and <b>Multi-Head Attention</b> introduced by <b><i>Vaswani et al.</i></b>
If not, you may read this blog, [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alamar.

<hr/>

<h1><center>Can we do better than GCNs?</center></h1>

From Graph Convolutional Network (GCN), we learnt that combining local graph structure and node-level features yields good performance on node classification task. However, the way GCN aggregates messages is <b>structure-dependent</b>, which may hurt its generalizability.

The fundamental novelty that GAT brings to the table is how the information from the one-hop neighborhood is aggregated. For GCN, a graph convolution operation produces the normalized sum of neighbors' node features as follows:

$$h_i^{(l+1)}=\sigma\left(\sum_{j\in \mathcal{N}(i)} {\frac{1}{c_{ij}} W^{(l)}h^{(l)}_j}\right)$$

where $\mathcal{N}(i)$ is the set of its one-hop neighbors (to include $v_{i}$ in the set, we simply added a self-loop to each node), $c_{ij}=\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}$ is a normalization constant based on graph structure, $\sigma$ is an activation function (GCN uses ReLU), and $W^{l}$ is a shared weight matrix for node-wise feature transformation.

GAT introduces the attention mechanism as a substitute for the statically normalized convolution operation. The figure below clearly illustrates the key difference.

<!-- {{< figure src="/images/blogs/GAT/GCN_vs_GAT.jpg" title="GCN vs GAT" width=800x class="floatcenter">}} -->
<center><strong>GCN vs GAT</strong></center>
<img src="/images/blogs/GAT/GCN_vs_GAT.jpg" width=800x/>

<hr/>

<!-- ## How does the attention work in GAT layer? -->
<h1><center>How does the GAT layer work?</center></h1>

The particular attentional setup utilized by GAT closely follows the work of `Bahdanau et al. (2015)` i.e <i>Additive Attention</i>, but the framework is agnostic to the particular choice of attention mechanism.

The input to the layer is a set of node features, $\mathbf{h} = \{\vec{h}_1,\vec{h}_2,...,\vec{h}_N\}, \vec{h}_i ∈ \mathbb{R}^{F}$ , where $N$ is the
number of nodes, and $F$ is the number of features in each node. The layer produces a new set of node
features (of potentially different cardinality $F'$ ), $\mathbf{h} = \{\vec{h'}_1,\vec{h'}_2,...,\vec{h'}_N\}, \vec{h'}_i ∈ \mathbb{R}^{F'}$, as its output.


<h3><font color="black" >The Attentional Layer broken into 4 separate parts:</font></h3>

<hr/>

<strong>1) <font color="red">Simple linear transformation:</font></strong> In order to obtain sufficient expressive power to transform the input features into higher level features, atleast one learnable linear transformation is required. To that end, as an initial step, a shared linear transformation, parametrized by a weight matrix, $W ∈ \mathbb{R}^{F′×F}$ , is applied to every node.

$$\begin{split}\begin{align}
z_i^{(l)}&=W^{(l)}h_i^{(l)} \\
\end{align}\end{split}$$

<img style="float: right;" src="/images/blogs/GAT/Attentional_Layer.jpg" width=400x/>

<hr/>

<span>
<strong>2) <font color="red">Attention Coefficients:</font> We then compute a pair-wise <font color="blue">un-normalized</font></strong>
</span> attention score between two neighbors. Here, it first concatenates the $z$ embeddings of the two nodes, where $||$ denotes concatenation, then takes a dot product of it with a learnable weight vector $\vec a^{(l)}$, and applies a LeakyReLU in the end. This form of attention is usually called additive attention, in contrast with the dot-product attention used for the Transformer model. We then perform self-attention on the nodes, a shared attentional mechanism $a$ : $\mathbb{R}^{F′} × \mathbb{R}^{F′} → \mathbb{R}$ to compute attention coefficients 
$$\begin{split}\begin{align}
e_{ij}^{(l)}&=\text{LeakyReLU}(\vec a^{(l)^T}(z_i^{(l)}||z_j^{(l)}))\\
\end{align}\end{split}$$

**Q. Is this step the most important step?** 

**Ans.** Yes! This indicates the importance of node $j’s$ features to node $i$. This step allows every node to attend on every other node, dropping all structural information.

**NOTE:** The graph structure is injected into the mechanism by performing <b>*masked attention*</b>, we only compute $e_{ij}$ for nodes $j$ ∈ $N_{i}$, where $N_{i}$ is some neighborhood of node $i$ in the graph. In all the experiments, these will be exactly the first-order neighbors of $i$ (including $i$).


<hr/>

<strong>3) <font color="red">Softmax:</font></strong> This makes coefficients easily comparable across different nodes, we normalize them across all choices of $j$ using the softmax function

$$\begin{split}\begin{align}
\alpha_{ij}^{(l)}&=\frac{\exp(e_{ij}^{(l)})}{\sum_{k\in \mathcal{N}(i)}^{}\exp(e_{ik}^{(l)})}\\
\end{align}\end{split}$$


<hr/>

<strong>4) <font color="red">Aggregation:</font></strong> This step is similar to GCN. The embeddings from neighbors are aggregated together, scaled by the attention scores. 

$$\begin{split}\begin{align}
h_i^{(l+1)}&=\sigma\left(\sum_{j\in \mathcal{N}(i)} {\alpha^{(l)}_{ij} z^{(l)}_j }\right)
\end{align}\end{split}$$

<hr/>

<!-- ![Attentional Layer](Attentional_Layer.jpg){:style="float: right;margin-right: 7px;margin-top: 7px;"} -->


<!-- ![Attentional Layer](Attentional_Layer.jpg =250x) -->


<!-- ### Multi-head Attention -->
<h2><center>Multi-head Attention</center></h2>

<figure>
  <img style="float: right; width: 70%" src="/images/blogs/GAT/MultiHead_Attention.jpeg">
  <figcaption>An illustration of multi-head attention (with K = 3 heads) by node 1 on its neighborhood. Different arrow styles and colors denote independent attention computations. The aggregated features from each head are concatenated or averaged to obtain $\vec{h'}_{1}$.</figcaption>
</figure>

<br clear=”both” />

Analogous to multiple channels in a Convolutional Net, GAT uses multi-head attention to enrich the model capacity and to stabilize the learning process. Specifically, K independent attention mechanisms execute the transformation of Equation 4, and then their outputs can be combined in 2 ways depending on the use:

$$\textbf{$ \color{red}{Average} $}: h_{i}^{(l+1)}=\sigma\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{k}W^{k}h^{(l)}_{j}\right)$$
$$\textbf{$ \color{green}{Concatenation} $}: h^{(l+1)}_{i}=||_{k=1}^{K}\sigma\left(\sum_{j\in \mathcal{N}(i)}\alpha_{ij}^{k}W^{k}h^{(l)}_{j}\right)$$

<b>1) <font color="green">Concatenation</font></b>
As can be seen in this setting, the final returned output, $h′$, will consist of $KF′$ features (rather than F′) for each node.

<b>2) <font color="red">Averaging</font></b>

If we perform multi-head attention on the final (prediction) layer of the network, concatenation is no longer sensible and instead, averaging is employed, and delay applying the final nonlinearity (usually a softmax or logistic sigmoid for classification problems).

<b>Thus <font color="green">concatenation for intermediary layers</font> and <font color="red">average for the final layer</font> are used.</b>

<hr/>

<!-- ## Implementing GAT Layer in PyTorch -->
<h1><center>Implementing GAT Layer in PyTorch</center></h1>

## Imports

```python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2020) # seed for reproducible numbers
```

## GAT Layer

{{< highlight python3 >}}
class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e       = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime   = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
{{< /highlight >}}

<!-- ## Implementing GAT on Citation Datasets using PyTorch Geometric -->
<h1><center>Implementing GAT on Citation Datasets using PyTorch Geometric</center></h1>

### PyG Imports

```python
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
%matplotlib notebook

import warnings
warnings.filterwarnings("ignore")
```

```python
name_data = 'Cora'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)
```

### Model

```python
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
```

### Train

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAT().to(device)

data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    if epoch%200 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()
```

### Evaluate

```python
model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
```

## References

[Graph Attention Networks](https://arxiv.org/abs/1710.10903)

[Graph attention network, DGL by Zhang et al.](https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html)

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[Mechanics of Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

[Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

### Written By

* Anirudh Dagar