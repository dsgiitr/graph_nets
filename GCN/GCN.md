---
title: "A Review : Graph Convolutional Networks (GCN)"
date: 2020-01-01T23:40:49+00:00
description : "Machine Learning / Graph Representation Learning"
type: post
image: images/blogs/GCN/gcn_architecture.png
author: Anirudh Dagar, Ajit Pant, Shubham Chandel, Shashank Gupta
tags: ["Graph Representation Learning"]
---

<h1><center><font color="green"> Introduction</font></center></h1>

<h2><font color="purple"> Graphs</font></h2>

Whom are we kidding! You may skip this section if you know what graphs are.

If you are here and haven't skipped this section, then, we assume that you are a complete beginner, you may want to read everything very carefully. We can define a graph as a picture that represents the data in an organised manner. Let's go deep into applied graph theory. A graph (being directed or undirected) consists of a set of vertices (or nodes) denoted by V, and a set of edges denoted by E. Edges can be weighted or binary. Let's have a look at a graph. 

<img src="/images/blogs/GCN/graph.png" width=800x/ height=400x/>

In the above graph we have:-

$$V = \\{A, B, C, D, E, F, G\\}$$

$$E = \\{(A,B), (B,C), (C,E), (B,D), (E,F), (D,E), (B,E), (G,E)\\}$$

Above all these edges, their corresponding weights have been specified. These weights can represent different quantities. For example, if we consider these nodes as different cities, edges can be the distance between these cities.

<hr/>


<h1><center><font color="green"> Terminology </font></center></h1>

You may skip this as well, if comfortable.


<img src="/images/blogs/GCN/Adjacency_Matrix.jpg" >

<ul>
 <li><font color="red"> <b>Node </b> </font>:<font color="blue"> A node is an entity in the graph. Here, represented by circles in the graph.</font></li>
 <li> <font color="red"> <b>Edge</b></font>:<font color="blue"> It is the line joining two nodes in a graph. The presence of an edge between two nodes represents the relationship between the nodes. Here, represented by straight lines in the graph.</font></li>
 <li> <font color="red"> <b>Degree of a vertex</b></font>:<font color="blue"> The degree of a vertex V of a graph G (denoted by deg (V)) is the number of edges incident with the vertex V. As an instance consider node B, it has 3 outgoing edges and 1 incoming edge, so outdegree is 3 and indegree is 1.</font></li>
 <li><font color="red">  <b>Adjacency Matrix</b></font>:<font color="blue"> It is a method of representing a graph using only a square Matrix. Suppose there are N nodes in a graph, then there will be N rows and N columns in the corresponding adjacency matrix. The ith row will contain a 1 in the jth column if there is an edge between the ith and the jth node; otherwise, it will contain a 0.</font></li>
</ul>
<h1><center><font color="green"> Why GCNs? </font></center></h1>

So let's get into the real deal. Looking around us, we can observe that most of the real-world datasets come in the form of graphs or networks: social networks, protein-interaction networks, the World Wide Web, etc. This makes learning on graphs an exciting problem that can solve tonnes of domain-specific tasks rendering us insightful information.

<b>But why can't these Graph Learning problems be solved by conventional Machine Learning/Deep Learning algorithms like CNNs? Why exactly was there a need for making a whole new class of networks?</b>

    A) To introduce a new XYZNet?
    B) To publish the 'said' novelty in a top tier conference?
    C) To you know what every other paper aims to achieve? 

No! No! No! Not because *Kipf and Welling* wanted to sound cool and publish yet another paper in a top tier conference. You see, not everything is an Alchemy :P. On that note, I'd suggest watching this super interesting, my favourite [talk](https://www.youtube.com/watch?v=x7psGHgatGM) by Ali Rahimi, which is really relevant today in the ML world.

So getting back to the topic, obviously I'm joking about these things, and surely this is a really nice contribution and GCNs are really powerful, Ok! Honestly, take the last part with a pinch of salt and <b><font color="red">remember</font></b> to ask me at the end.

<hr/>

<b>But I still haven't answered the big elephant in the room. WHY?</b> 

To answer why, we first need to understand how a class of models like Convolutional Neural Networks(CNNs) work. CNN's are really powerful, and they have the capacity to learn very high dimensional data. Say you have a $512*512$ pixel image. The dimensionality here is approximately 1 million. For 10 samples, the space becomes $10^{1,000,000}$, and CNNs have proven to work really well on such tough task settings! 

But there is a catch! These data samples, like images, videos, audio, etc., where CNN models are mostly used, all have a specific compositionality, which is one of the strong assumptions we made before using CNNs. 

So CNNs basically extract the compositional features and feeds them to the classifier.
<hr/>

<b>What do I mean by compositionality?</b>

The key properties of the assumption of compositionality are
<ol>
<li><font color="green">Locality</font></li>

<li><font color="green"> Stationarity or Translation Invariance</font></li>    

<li><font color="green"> Multi-Scale: Learning Hierarchies of representations</font></li>
</ol>
<hr/>

<b>2D Convolution vs. Graph Convolution</b>

If you haven't figured it out, not all types of data lie on the Euclidean Space and such are the graphs data types, including manifolds, and 3D objects, thus rendering the previous 2D Convolution useless. Hence, the need for GCNs which have the ability to capture the inherent structure and topology of the given graph. Hence this blog :P. 

<left><img src="/images/blogs/GCN/CNN_to_GCN.jpg" ></left>

<h1><center><font color="green"> Appllications of GCNs  </font></center></h1>

One possible application of GCN is in the Facebook's friend prediction algorithm. Consider three people <i>A</i>, <i>B</i> and <i>C</i>. Given that <i>A</i> is a friend of <i>B</i>, <i>B</i> is a friend of <i>C</i>. You may also have some representative information in the form of features about each person, for example, <i>A</i> may like movies starring Liam Neeson and in general <i>C</i> is a fan of genre Thriller, now you have to predict whether <i>A</i> is friend of <i>C</i>.


<figure >
   <center> <img src="/images/blogs/GCN/GCN_FB_Link_Prediction_Social_Nets.jpg"></center>
   <center> <figcaption>Facebook Link Prediction for Suggesting Friends using Social Networks</figcaption></center>
</figure>


<h1><center><font color="green">What GCNs? </font></center></h1>

As the name suggests, Graph Convolution Networks (GCNs), draw on the idea of Convolution Neural Networks re-defining them for the non-euclidean data domain. A regular Convolutional Neural Network used popularly for Image Recognition, captures the surrounding information of each pixel of an image. Similar to euclidean data like images, the convolution framework here aims to capture neighbourhood information for non-euclidean spaces like graph nodes.

A GCN is basically a neural network that operates on a graph. It will take a graph as an input and give some (we'll see what exactly) meaningful output.

<b>GCNs come in two different styles</b>: 

<ul>
 <li> <b>Spectral GCNs</b>: Spectral-based approaches define graph convolutions by introducing filters from the perspective of graph signal processing based on graph spectral theory.</li>
 <li> <b>Spatial GCNs</b>: Spatial-based approaches formulate graph convolutions as aggregating feature information from neighbours.</li>
</ul>

Note: Spectral approach has the limitation that all the graph samples must have the same structure, i.e. homogeneous structure. But it is a hard constraint, as most of the real-world graph data have different structure and size for different samples i.e. heterogeneous structure. The spatial approach is agnostic of the graph structure.

<h1><center><font color="green"> How GCNs?  </font></center></h1>

First, let's work this out for the Friend Prediction problem and then we will generalize the approach.

<b>Problem Statement</b>: You are given N people and also a graph where there is an edge between two people if they are friends. You need to predict whether two people will become friends in the future or not.

A simple graph corresponding to this problem is:

<img src="/images/blogs/GCN/friends_graph.png" >

Here person $(1,2)$ are friends, similarly $(2,3), (3,4), (4,1), (5,6), (6,8), (8,7), (7,6)$ are also friends.

Now we are interested in finding out whether a given pair of people are likely to become friends in the future or not. Let's say that the pair we are interested in is $(1,3)$, and now since they have 2 common friends, we can softly imply they have a chance of becoming friends, whereas the nodes $(1,5)$ have no friend in common, so they are less likely to become friends.

Let's take another example:
<img src="/images/blogs/GCN/friends_graph2.png" >

Here $(1,11)$ are much more likely to become friends than say $(3, 11)$.


Now the question that one can raise is 'How to implement and achieve this result?'. GCN's implement it in a way similar to CNNs. In a CNN, we apply a filter on the original image to get the representation in the next layer. Similarly, in GCN, we apply a filter which creates the next layer representation. 

Mathematically we can define as follows: $$H^{i} = f(H^{i-1}, A)$$


A very simple example of $f$ maybe:

$$f(H^{i}, A) = σ(AH^{i}W^{i})$$



where
 - $A$ is the $N × N$ adjacency matrix
 - $X$ is the input feature matrix $N × F$, where $N$ is the number of nodes and $F$ is the number of input features for each node.
 - $σ$ is the Relu activation function
 - $H^{0} = X$ Each layer $H^{i}$ corresponds to an $N × F^{i}$ feature matrix where each row is a feature representation of a node.
 - $f$ is the propagation rule
 
At each layer, these features are aggregated to form the next layer’s features using the propagation rule $f$. In this way, features become increasingly more abstract at each consecutive layer.


Yes, that is it, we already have some function to propagate information across the graphs which can be trained in a semi-supervised way. Using the GCN layer, the representation of each node (each row) is now a sum of its neighbour's features! In other words, the layer represents each node as an aggregate of its neighbourhood.

<b>But, Wait is it so simple?</b>

I'll request you to stop for a moment here and think really hard about the function we just defined.

Is that correct?

<b>STOP</b>

....

....

....


It is sort of! But it is not exactly what we want. If you were unable to arrive at the problem, fret not. Let's see what exactly are the <b><font color="red">'problems'</font></b> (yes, more than one problem) this function might lead to:

<b><font color="red">1. The new node features $H^{i}$ are not a function of its previous representation</b></font>: As you might have noticed, the aggregated representation of a node is only a function of its neighbours and does not include its own features. If not handled, this may lead to the loss of the node identity and hence rendering the feature representations useless. We can easily fix this by adding self-loops, that is an edge starting and ending on the same node; in this way, a node will become a neighbour of itself. Mathematically, self-loops are nothing but can be expressed by adding the identity matrix to the adjacency matrix.


<b><font color="red">2. Degree of the nodes lead to the values being scaled asymmetrically across the graph</b></font>: In simple words, nodes that have a large number of neighbours (higher degree) will get much more input in the form of neighbourhood aggregation from the adjacent nodes and hence will have a larger value and vice versa may be true for nodes with smaller degrees having small values. This can lead to problems during the training of the network. To deal with the issue, we will be using normalisation, i.e., reduce all values in such a way that the values are on the same scale. Normalising $A$ such that all rows sum to one, i.e. $D^{−1}A$, where $D$ is the diagonal node degree matrix, gets rid of this problem. Multiplying with $D^{−1}A$ now corresponds to taking the average of neighboring node features. According to the authors, after observing empirical results, they suggest "In practice, dynamics get more interesting when we use symmetric normalisation, i.e. $\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$ (as this no longer amounts to mere averaging of neighbouring nodes).
 
</ul>
After addressing the two problems stated above, the new propagation function $f$ is:

$$f(H^{(l)}, A) = \sigma\left( \hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$


where
 - $\hat{A} = A + I$
 - $I$ is the identity matrix
 - $\hat{D}$  is the diagonal node degree matrix of $\hat{A}$.

<hr/>

<br/>

<h1><center><font color="green">Implementing GCNs from Scratch in PyTorch</font></center></h1>

<br/>

We are now ready to put all of the tools together to deploy our very first fully-functional Graph Convolutional Network. In this tutorial, we will be training GCN on the 'Zachary Karate Club Network'. We will be using the '[Semi Supervised Graph Learning Model](https://arxiv.org/abs/1609.02907)' proposed in the paper by <b>Thomas Kipf & Max Welling</b>.


<center><h2><u> Zachary Karate Club </u></h2></center>

During the period from 1970-1972, Wayne W. Zachary observed the people belonging to a local karate club. He represented these people as nodes in a graph. And added an edge between a pair of people if they interacted with each other. The result was the graph shown below.

<center><img src="/images/blogs/GCN/karate_club.png" width="800x"></center>


During the study, an interesting event happened. A conflict arose between the administrator "John A" and instructor "Mr. Hi" (pseudonyms), which led to the split of the club into two. Half of the members formed a new club around Mr. Hi; members from the other part found a new instructor or gave up karate. 

Using the graph that he had found earlier, he tried to predict which member will go to which half. And surprisingly he was able to predict the decision of all the members except for node 9 who went with Mr. Hi instead of John A.

Zachary used the maximum flow – minimum cut Ford–Fulkerson algorithm for this. We will be using a different algorithm today; hence it is not required to know about the Ford-Fulkerson algorithm.

Here we will be using the Semi-Supervised Graph Learning Method. Semi-Supervised means that we have labels for only some of the nodes, and we have to find the labels for other nodes. Like in this example we have the labels for only the nodes belonging to 'John A' and 'Mr. Hi', we have not been provided with labels for any other member, and we have to predict that only on the basis of the graph given to us.

<h1><center><font color="green"> Required Imports </font></center></h1>

In this post, we will be using PyTorch and Matplotlib.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
```
<br/>

<h1><center><font color="green"> The Convolutional Layer  </font></center></h1>

First, we will be creating the GCNConv class, which will serve as the Layer creation class. Every instance of this class will be getting Adjacency Matrix as input and will be outputting 'RELU(A_hat * X * W)', which the Net class will use.


```python
class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A+torch.eye(A.size(0))
        self.D     = torch.diag(torch.sum(A,1))
        self.D     = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W     = nn.Parameter(torch.rand(in_channels,out_channels))
    
    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return out

class Net(torch.nn.Module):
    def __init__(self,A, nfeat, nhid, nout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid)
        self.conv2 = GCNConv(A,nhid, nout)
        
    def forward(self,X):
        H  = self.conv1(X)
        H2 = self.conv2(H)
        return H2
```

<br/>

```python
# 'A' is the adjacency matrix, it contains 1 at a position (i,j)
# if there is a edge between the node i and node j.
A = torch.Tensor([[0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                [1,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
                [1,1,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0],
                [1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1],
                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1],
                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1],
                [0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,1],
                [0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,0,0,0,0,1,1,1,0,1],
                [0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,0]
                ])

target=torch.tensor([0,-1,-1,-1, -1, -1, -1, -1,-1,-1,-1,-1, -1, -1, -1, -1,-1,-1,-1,-1, -1, -1, -1, -1,-1,-1,-1,-1, -1, -1, -1, -1,-1,1])
```

In this example, we have the label for admin(node 1) and instructor(node 34) so only these two contain the class label(0 and 1) all other are set to -1, which means that the predicted value of these nodes will be ignored in the computation of loss function.


X is the feature matrix. Since we don't have any feature of each node, we will just be using the one-hot encoding corresponding to the index of the node.


<h1><center><font color="green"> Training </font></center></h1>

```python
X=torch.eye(A.size(0))

# Here we are creating a Network with 10 features
# in the hidden layer and 2 in the output layer.

T=Net(A,X.size(0), 10, 2)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(T.parameters(), lr=0.01, momentum=0.9)

loss=criterion(T(X),target)


# Plot animation using celluloid

fig = plt.figure()
camera = Camera(fig)

for i in range(200):
    optimizer.zero_grad()
    loss=criterion(T(X), target)
    loss.backward()
    optimizer.step()
    l=(T(X));

    plt.scatter(l.detach().numpy()[:,0],l.detach().numpy()[:,1],
        c=[0, 0, 0, 0 ,0 ,0 ,0, 0, 1, 1, 0 ,0, 0, 0, 1 ,1 ,0 ,0
        ,1, 0, 1, 0 ,1 ,1, 1, 1, 1 ,1 ,1, 1, 1, 1, 1, 1 ])
    
    for i in range(l.shape[0]):
        text_plot = plt.text(l[i,0], l[i,1], str(i+1))

    camera.snap()

    if i%20==0:
        print("Cross Entropy Loss: =", loss.item())

animation = camera.animate(blit=False, interval=150)
animation.save('./train_karate_animation.mp4', writer='ffmpeg', fps=60)
HTML(animation.to_html5_video())
```


<center><img src="/images/blogs/GCN/train_karate_animation.gif"></center>


As you can see above, it has divided the data into two categories, and it is close to what happened to reality. 
<hr/>

<h1><center><font color="green"> PyTorch Geometric Implementation </font></center></h1>

We also implemented GCNs using this great library [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric/) (PyG) with a super active maintainer [Matthias Fey](https://github.com/rusty1s/). PyG is specifically built for PyTorch lovers who need an easy, fast and simple way out to implement and test their work on various Graph Representation Learning papers.

You can find our implementation made using PyTorch Geometric in the following notebook <b><font color="red">[GCN_PyG Notebook](https://github.com/dsgiitr/graph_nets/blob/master/GCN/GCN_PyG.ipynb)</font></b> with GCN trained on a Citation Network, the Cora Dataset. Also all the code used in the blog along with IPython notebooks can be found at the github repository <b>[graph_nets](https://github.com/dsgiitr/graph_nets)</b>.

<hr/>


<h1> References </h1>
We strongly recommend reading up these references as well to make your understanding solid. 

<strong>Also, remember we asked you to remember one thing? To answer that read up on this amazing blog which tries to understand if GCNs really are powerful as they claim to be. [How powerful are Graph Convolutions?](https://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2/)</strong>

* [Code & GitHub Repository](https://github.com/dsgiitr/graph_nets)

* [Blog GCNs by Thomas Kipf](https://tkipf.github.io/graph-convolutional-networks/)

* [Semi-Supervised Classification with Graph Convolutional Networks by Thomas Kipf and Max Welling](https://arxiv.org/abs/1609.02907)

* [How to do Deep Learning on Graphs with Graph Convolutional Networks by Tobias Skovgaard Jepsen](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780)

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

<hr/>


<h1> Written By </h1>
<ul>

<li> Ajit Pant</li>
<li> Shubham Chandel</li>
<li> Anirudh Dagar</li>
<li> Shashank Gupta</li>
