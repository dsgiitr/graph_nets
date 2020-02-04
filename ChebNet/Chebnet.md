---
title: "ChebNet: CNN on Graphs with Fast Localized Spectral Filtering"
date: 2020-02-01T23:40:49+00:00
description : "Machine Learning / Graph Representation Learning"
type: post
image: https://storage.googleapis.com/groundai-web-prod/media/users/user_3036/project_14426/images/x1.png
author: Shashank Gupta
tags: ["Graph Representation Learning"]
---

<h2><center>Motivation</center></h2>

As a part of this blog series, this time we'll be looking at a spectral convolution technique introduced in the paper by M. Defferrard, X. Bresson, and P. Vandergheynst, on "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering".

<br/>

As mentioned in our previous blog on [A Review : Graph Convolutional Networks (GCN)](https://dsgiitr.com/blogs/gcn/), the spatial convolution and pooling operations are well-defined only for the Euclidean domain. Hence, we cannot apply the convolution directly on the irregular structured data such as graphs.

The technique proposed in this paper provide us with a way to perform convolution on graph like data, for which they used convolution theorem. According to which, Convolution in spatial domain is equivalent to multiplication in Fourier domain. Hence, instead of performing convolution explicitly in the spatial domain, we will transform the graph data and the filter into Fourier domain. Do element-wise multiplication and the result is converted back to spatial domain by performing inverse Fourier transform. Following figure illustrates the proposed technique:

<center><img src="/images/blogs/ChebNet/fft.jpg"></center>
<hr/>

<h2><center>But How to Take This Fourier Transform?</center></h2>

As mentioned we have to take a fourier transform of graph signal. In spectral graph theory, the important operator used for Fourier analysis of graph is the Laplacian operator. For the graph $G=(V,E)$, with set of vertices $V$ of size $n$ and set of edges $E$. The Laplacian is given by

$$Δ=D−A$$

where $D$ denotes the diagonal degree matrix and $A$ denotes the adjacency matrix of the graph.

When we do eigen-decomposition of the Laplacian, we get the orthonormal eigenvectors, as the Laplacian is real symmetric positive semi-definite matrix (side note: positive semidefinite matrices have orthogonal eigenvectors and symmetric matrix has real eigenvalues). These eigenvectors are denoted by $\{ϕ_l\}^n_{l=0}$ and also called as Fourier modes. The corresponding eigenvalues $\{λ_l\}^n_{l=0}$ acts as frequencies of the graph.

The Laplacian can be diagonalized by the Fourier basis.

$$Δ=ΦΛΦ^T$$

where, $Φ=\{ϕ_l\}^n_{l=0}$ is a matrix with eigenvectors as columns and $Λ$ is a diagonal matrix of eigenvalues.

Now the graph can be transformed to Fourier domain just by multiplying by the Fourier basis. Hence, the Fourier transform of graph signal $x:V→R$ which is defined on nodes of the graph $x∈R^n$ is given by:

$\hat{x}=Φ^Tx$, where $\hat{x}$ denotes the graph Fourier transform. Hence, the task of transforming the graph signal to Fourier domain is nothing but the matrix-vector multiplication.<br>

Similarly, the inverse graph Fourier transform is given by:<br>
$x=Φ\hat{x}$.<br>
This formulation of Fourier transform on graph gives us the required tools to perform convolution on graphs. 

<hr/>

<h2><center>Filtering of Signals on Graph</center></h2>

As we now have the two necessary tools to define convolution on non-Euclidean domain:

1) Way to transform graph to Fourier domain.

2) Convolution in Fourier domain, the convolution operation between graph signal $x$ and filter $g$ is given by the graph convolution of the input signal $x$ with a filter $g∈R^n$ defined as:


$x∗_Gg=ℱ^{−1}(ℱ(x)⊙ℱ(g))=Φ(Φ^Tx⊙Φ^Tg)$,


where $⊙$ denotes  the  element-wise  product.  If  we  denote  a filter as $g_θ=diag(Φ^Tg)$, then the spectral graph convolution is simplified as $x∗_Gg_θ=Φg_θΦ^Tx$

<hr/>
<h2><center>Why can't we go forward with this scheme?</center></h2>

All spectral-based ConvGNNs follow this definition. But, this method has three major problems:

1. The number of filter parameters to learn depends on the dimensionality of the input which translates into O(n) complexity and filter is non-parametric.

2. The filters are not localized i.e. filters learnt for graph considers the entire graph, unlike traditional CNN which takes only nearby local pixels to compute convolution.

3. The algorithm needs to calculate the eigen-decomposition explicitly and multiply signal with Fourier basis as there is no Fast Fourier Transform algorithm defined for graphs, hence the computation is $O(n^2)$. (Fast Fourier Transform defined for Euclidean data has $O(nlogn)$ complexity)

<hr/>
<h2><center>Polynomial Parametrization of Filters</center></h2>

To overcome these problems they used an polynomial approximation to parametrize the filter.<br>
Now, filter is of the form of:<br>
$g_θ(Λ) =\sum_{k=0}^{K-1}θ_kΛ_k$, where the parameter $θ∈R^K$ is a vector of polynomial coefficients.<br>
These spectral filters represented by $Kth$-order polynomials of the Laplacian are exactly $K$-localized.  Besides, their learning complexity is $O(K)$, the support size of the filter, and thus the same complexity as classical CNNs.

<hr/>
<h2><center>Is everything fixed now?</center></h2>

No, the cost to filter a signal is still high with $O(n^2)$ operations because of the multiplication with the Fourier basis U. (calculating the eigen-decomposition explicitly and multiply signal with Fourier basis)

To bypass this problem, the authors parametrize $g_θ(Δ)$ as a polynomial function that can be computed recursively from $Δ$. One  such  polynomial,  traditionally  used in  Graph Signal Processing to  approximate  kernels,  is  the  <b>Chebyshev  expansion</b>. The Chebyshev polynomial $T_k(x)$ of order $k$ may be computed by the stable recurrence relation $T_k(x) = 2xT_{k−1}(x)−T_{k−2}(x)$ with $T_0=1$ and $T_1=x$.

The spectral filter is now given by a truncated Chebyshev polynomial:

$$g_θ(\barΔ)=Φg(\barΛ)Φ^T=\sum_{k=0}^{K-1}θ_kT_k(\barΔ)$$

where, $Θ∈R^K$ now represents a vector of the Chebyshev coefficients, the $\barΔ$ denotes the rescaled $Δ$. (This rescaling is necessary as the Chebyshev polynomial form orthonormal basis in the interval [-1,1] and the eigenvalues of original Laplacian lies in the interval $[0,λ_{max}]$). Scaling is done as  $\barΔ= 2Δ/λ_{max}−I_n$.

The filtering operation can now be written as $y=g_θ(Δ)x=\sum_{k=0}^{K-1}θ_kT_k(\barΔ)x$, where, $x_{i,k}$ are the input feature maps, $Θ_k$ are the trainable parameters.

<hr/>

<h2><center>Pooling Operation</center></h2>

In case of images, the pooling operation consists of taking a fixed size patch of pixels, say 2x2, and keeping only the pixel with max value (assuming you apply max pooling) and discarding the other pixels from the patch. Similar concept of pooling can be applied to graphs.

Defferrard  et  al.  address  this  issue  by using the coarsening phase of the Graclus multilevel clustering algorithm. Graclus’ greedy rule consists, at each coarsening level, in picking an unmarked vertex $i$ and matching it with one of its unmarked neighbors $j$ that maximizes the local normalized cut $Wij(1/di+ 1/dj)$.  The two matched vertices are then marked and the coarsened weights are set as the sum of their weights.  The matching is repeated until all nodes have been explored. This is an very fast coarsening scheme which divides the number of nodes by approximately two from one level to the next coarser level. After coarsening, the nodes of the input graph and its coarsened version are rearranged into a balanced binary tree.  Arbitrarily  aggregating  the  balanced  binary  tree  from bottom to top will arrange similar nodes together. Pooling such a  rearranged  signal  is  much  more  efficient  than  pooling  the original. The following figure shows the example of graph coarsening and pooling.

<center><img src="/images/blogs/ChebNet/pool.png"></center>

<hr/>

<h1><center>Implementing ChebNET in PyTorch</center></h1>


```python
## Imports

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import collections
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import sys

import os
```


```python
if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)
```

    cuda available


## Data Prepration


```python
# load data in folder datasets
mnist = input_data.read_data_sets('datasets', one_hot=False)

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels
```


```python
from grid_graph import grid_graph
from coarsening import coarsen
from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L

# Construct graph
t_start = time.time()
grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid
```

    nb edges:  6396



```python
# Compute coarsened graphs
coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)
```

    Heavy Edge Matching coarsening with Xavier version
    Layer 0: M_0 = |V| = 976 nodes (192 added), |E| = 3198 edges
    Layer 1: M_1 = |V| = 488 nodes (83 added), |E| = 1619 edges
    Layer 2: M_2 = |V| = 244 nodes (29 added), |E| = 794 edges
    Layer 3: M_3 = |V| = 122 nodes (7 added), |E| = 396 edges
    Layer 4: M_4 = |V| = 61 nodes (0 added), |E| = 194 edges



```python
# Compute max eigenvalue of graph Laplacians
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))
```

    lmax: [1.3857538, 1.3440963, 1.1994357, 1.0239158]



```python
# Reindex nodes to satisfy a binary tree structure
train_data = perm_data(train_data, perm)
val_data = perm_data(val_data, perm)
test_data = perm_data(test_data, perm)

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

print('Execution time: {:.2f}s'.format(time.time() - t_start))
del perm
```

    (55000, 976)
    (5000, 976)
    (10000, 976)
    Execution time: 4.18s


## Model


```python
# class definitions

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables, 
    called "my_sparse_mm", by subclassing torch.autograd.Function 
    and implementing the forward and backward passes.
    """
    
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y
    
    def backward(self, grad_output):
        W, x = self.saved_tensors 
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t()) 
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx
    
    
class Graph_ConvNet_LeNet5(nn.Module):
    
    def __init__(self, net_parameters):
        
        print('Graph ConvNet: LeNet5')
        
        super(Graph_ConvNet_LeNet5, self).__init__()
        
        # parameters
        D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        FC1Fin = CL2_F*(D//16)
        
        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F) 
        Fin = CL1_K; Fout = CL1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K; self.CL1_F = CL1_F; 
        
        # graph CL2
        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F) 
        Fin = CL2_K*CL1_F; Fout = CL2_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl2.weight.data.uniform_(-scale, scale)
        self.cl2.bias.data.fill_(0.0)
        self.CL2_K = CL2_K; self.CL2_F = CL2_F; 

        # FC1
        self.fc1 = nn.Linear(FC1Fin, FC1_F) 
        Fin = FC1Fin; Fout = FC1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin
        
        # FC2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F; Fout = FC2_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # nb of parameters
        nb_param = CL1_K* CL1_F + CL1_F          # CL1
        nb_param += CL2_K* CL1_F* CL2_F + CL2_F  # CL2
        nb_param += FC1Fin* FC1_F + FC1_F        # FC1
        nb_param += FC1_F* FC2_F + FC2_F         # FC2
        print('nb of parameters=',nb_param,'\n')
        
        
    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( 2.0/ (Fin+Fout) )
        W.uniform_(-scale, scale)

        return W
        
        
    def graph_conv_cheby(self, x, cl, L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin) 

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax) 
        
        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T 
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data) 
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable( L , requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()
        
        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B
        
        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B  
             
        if K > 1: 
            x1 = my_sparse_mm()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L,x1) - x0  
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B
            x0, x1 = x1, x2  
        
        x = x.view([K, V, Fin, B])           # K x V x Fin x B     
        x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K       
        x = x.view([B*V, Fin*K])             # B*V x Fin*K
        
        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout  
        x = x.view([B, V, Fout])             # B x V x Fout
        
        return x
        
        
    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p): 
        if p > 1: 
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p          
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x  
        else:
            return x    
        
        
    def forward(self, x, d, L, lmax):
        
        # graph CL1
        x = x.unsqueeze(2) # B x V x Fin=1  
        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)
        
        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)
        
        # FC1
        x = x.view(-1, self.FC1Fin)
        x = self.fc1(x)
        x = F.relu(x)
        x  = nn.Dropout(d)(x)
        
        # FC2
        x = self.fc2(x)
            
        return x
        
        
    def loss(self, y, y_target, l2_regularization):
    
        loss = nn.CrossEntropyLoss()(y,y_target)

        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()
           
        loss += 0.5* l2_regularization* l2_loss
            
        return loss
    
    
    def update(self, lr):
                
        update = torch.optim.SGD( self.parameters(), lr=lr, momentum=0.9 )
        
        return update
        
        
    def update_learning_rate(self, optimizer, lr):
   
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    
    def evaluation(self, y_predicted, test_l):
    
        _, class_predicted = torch.max(y_predicted.data, 1)
        return 100.0* (class_predicted == test_l).sum()/ y_predicted.size(0)

```


```python
# network parameters
D = train_data.shape[1]
CL1_F = 32
CL1_K = 25
CL2_F = 64
CL2_K = 25
FC1_F = 512
FC2_F = 10
net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]
```


```python
# instantiate the object net of the class 
net = Graph_ConvNet_LeNet5(net_parameters)
if torch.cuda.is_available():
    net.cuda()
print(net)
```

    Graph ConvNet: LeNet5
    nb of parameters= 2056586 
    
    Graph_ConvNet_LeNet5(
      (cl1): Linear(in_features=25, out_features=32, bias=True)
      (cl2): Linear(in_features=800, out_features=64, bias=True)
      (fc1): Linear(in_features=3904, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=10, bias=True)
    )



```python
# Weights
L_net = list(net.parameters())
```

## Hyper parameters  setting


```python
# learning parameters
learning_rate = 0.05
dropout_value = 0.5
l2_regularization = 5e-4 
batch_size = 100
num_epochs = 20
train_size = train_data.shape[0]
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=',num_epochs,', train_size=',train_size,', nb_iter=',nb_iter)
```

    num_epochs= 20 , train_size= 55000 , nb_iter= 11000


## Training & Evaluation 


```python
# Optimizer
global_lr = learning_rate
global_step = 0
decay = 0.95
decay_steps = train_size
lr = learning_rate
optimizer = net.update(lr) 


# loop over epochs
indices = collections.deque()
for epoch in range(num_epochs):  # loop over the dataset multiple times

    # reshuffle 
    indices.extend(np.random.permutation(train_size)) # rand permutation
    
    # reset time
    t_start = time.time()
    
    # extract batches
    running_loss = 0.0
    running_accuray = 0
    running_total = 0
    while len(indices) >= batch_size:
        
        # extract batches
        batch_idx = [indices.popleft() for i in range(batch_size)]
        train_x, train_y = train_data[batch_idx,:], train_labels[batch_idx]
        train_x = Variable( torch.FloatTensor(train_x).type(dtypeFloat) , requires_grad=False) 
        train_y = train_y.astype(np.int64)
        train_y = torch.LongTensor(train_y).type(dtypeLong)
        train_y = Variable( train_y , requires_grad=False) 
            
        # Forward 
        y = net.forward(train_x, dropout_value, L, lmax)
        loss = net.loss(y,train_y,l2_regularization) 
        loss_train = loss.data
        
        # Accuracy
        acc_train = net.evaluation(y,train_y.data)
        
        # backward
        loss.backward()
        
        # Update 
        global_step += batch_size # to update learning rate
        optimizer.step()
        optimizer.zero_grad()
        
        # loss, accuracy
        running_loss += loss_train
        running_accuray += acc_train
        running_total += 1
        
        # print        
        if not running_total%100: # print every x mini-batches
            print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch+1, running_total, loss_train, acc_train))
          
       
    # print 
    t_stop = time.time() - t_start
    print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' % 
          (epoch+1, running_loss/running_total, running_accuray/running_total, t_stop, lr))
 

    # update learning rate 
    lr = global_lr * pow( decay , float(global_step// decay_steps) )
    optimizer = net.update_learning_rate(optimizer, lr)
    
    
    # Test set
    running_accuray_test = 0
    running_total_test = 0
    indices_test = collections.deque()
    indices_test.extend(range(test_data.shape[0]))
    t_start_test = time.time()
    while len(indices_test) >= batch_size:
        batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
        test_x, test_y = test_data[batch_idx_test,:], test_labels[batch_idx_test]
        test_x = Variable( torch.FloatTensor(test_x).type(dtypeFloat) , requires_grad=False) 
        y = net.forward(test_x, 0.0, L, lmax) 
        test_y = test_y.astype(np.int64)
        test_y = torch.LongTensor(test_y).type(dtypeLong)
        test_y = Variable( test_y , requires_grad=False) 
        acc_test = net.evaluation(y,test_y.data)
        running_accuray_test += acc_test
        running_total_test += 1
    t_stop_test = time.time() - t_start_test
    print('  accuracy(test) = %.3f %%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))
```
<hr/>

You can find our implementation made using PyTorch at <b><font color="red">[ChebNet](https://github.com/dsgiitr/graph_nets/blob/master/ChebNet/Chebnet_Blog+Code.ipynb)</font></b>.

## References

- [Code & GitHub Repository](https://github.com/dsgiitr/graph_nets)

- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)

- [Xavier Bresson: "Convolutional Neural Networks on Graphs"](https://www.youtube.com/watch?v=v3jZRkvIOIM)