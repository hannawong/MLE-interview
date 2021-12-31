

# GraphSage 

### 1. GCN的局限

GCN本身有一个巨大局限，即没法快速表示**新节点**。GCN需要把**所有节点都参与训练**才能得到node embedding，如果新node来了，没法得到新node的embedding。而GraphSAGE使用**Inductive Learning**，解决了这个问题。

**得到新节点的表示的难处：**

要想得到新节点的表示，需要让新的node或者subgraph去和已经优化好的node embedding去“对齐”。然而每个节点的表示都是受到其他节点的**影响**（牵一发而动全身），因此添加一个节点，意味着**许许多多与之相关的节点的表示都应该调整**。

**因此我们需要换一种思路：**

既然**新增的节点，一定会改变原有节点的表示**，那么我们**干嘛一定要得到每个节点的一个固定的表示呢？**我们何不直接**学习一种节点的表示方法**。这样不管graph怎么改变，都可以很容易地得到新的表示。



### 2. GraphSAGE

针对这种问题，GraphSAGE模型提出了一种算法框架，可以很方便地得到新node的表示。

#### 2.1 . Embedding generation（前向传播算法）

Embedding generation算法共聚合K次，总共有K个聚合函数(aggregator)，可以认为是K层，来聚合邻居节点的信息。假如用$h^k$来表示第k层每个节点的embedding，那么如何从$h^{k-1}$得到$h^k$呢？

- $h^{0}$就是初始的每个节点embedding。

- 对于每个节点v，都把它随机采样的若干**邻居**的**k-1**层的所有向量表示$\{h^{k-1}_u, u \in N(v)\}$、以及节点v**自己**的k-1层表示聚合成一个向量，这样就得到了第$k$层的表示$h^k$。这个聚合方法具体是怎么做的后面再详细介绍。

文中描述如下：

![1d3847bc19cca879d10b78ced44c0dc](C:\Users\zh-wa\AppData\Local\Temp\WeChat Files\1d3847bc19cca879d10b78ced44c0dc.jpg)

下图简明的展现了上述过程：



![img](https://pic2.zhimg.com/80/v2-9e2b7329c0694eae4b3fdc1f224e6705_1440w.jpg)

这里需要注意的是，每一层的node的表示都是由**上一层**生成的，跟本层的其他节点无关。

随着层数K的增加，**可以聚合越来越远距离的信息**。虽然每次选择邻居的时候就是从周围的**一阶邻居**中均匀地采样固定个数个邻居，但是由于节点的邻居也聚合了其邻居的信息，这样，在下一次聚合时，该节点就会接收到其邻居的邻居的信息，也就是聚合到了**二阶邻居**的信息了。这就像社交图谱中“朋友的朋友”的概念。

![img](https://pic1.zhimg.com/80/v2-899c3f911296535889a29de8471582ac_1440w.jpg)



#### 2.2 聚合函数选择

- Mean Pooling: ![img](https://pic4.zhimg.com/80/v2-beaaa5540cc41f5936d23f704d403dd3_1440w.png)这个比较好理解，就是当前**节点v本身和它所有的邻居**在k-1层的embedding的mean。

- LSTM Aggregator：把当前节点v的邻居随机打乱，输入到LSTM中。作者的想法是说LSTM的模型capacity更强。但是节点周围的邻居明明是没有顺序的，这样做似乎有不妥。

- Pooling Aggregator：![img](https://pic3.zhimg.com/80/v2-4e3693bd199e660e3159d2ac0d58555a_1440w.png)

  把节点v的所有邻居节点都单独经过一个MLP+sigmoid得到一个向量，最后把所有邻居的向量做一个element-wise的max-pooling。

#### 2.3 GraphSAGE的参数学习

GraphSAGE的参数就是聚合函数的参数。为了学习这些参数，需要设计合适的损失函数。

对于**无监督学习**，设计的损失函数应该**让临近的节点的拥有相似的表示**，反之应该表示大不相同。所以损失函数是这样的：

![img](https://pic1.zhimg.com/80/v2-9c473f5e242f8db158854d4e5e036b9c_1440w.png)

其中，节点v是和节点u在一定长度的random walk上共现的节点，所以它们的点积要尽可能大；后面这项是采了Q个负样本，它们的点积要尽可能小。这个loss和skip-gram中的negative sampling如出一辙。

对于**有监督学习**，可以直接使用cross-entropy loss等常规损失函数。当然，上面的这个loss也可以作为一个辅助loss。



### 3. 和GCN的关系

原始GCN的方法，其实和GraphSAGE的Mean Pooling聚合方法是类似的，即每一层都聚合自己和自己邻居的归一化embedding表示。而GraphSAGE使用了其他capacity更大的聚合函数而已。

此外，GCN是一口气把整个图都丢进去训练，但是来了一个新的节点就不免又要把整个图重新训一次。而GraphSAGE则是在**增加了新的节点之后，来增量更新旧的节点，调整整张图的embedding表示**。只是生成新节点embedding的过程，实施起来相比于GCN更加灵活方便了。

----

参考：

https://zhuanlan.zhihu.com/p/74242097

