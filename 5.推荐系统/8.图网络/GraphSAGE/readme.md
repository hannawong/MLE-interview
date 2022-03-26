

# GraphSage (Graph Sample and Aggregate)

### 1. GCN的局限

GCN本身有一个巨大局限，即没法快速表示**新节点**。GCN需要把**所有节点都参与训练**才能得到node embedding(巨大的拉普拉斯矩阵)，如果新node来了，没法得到新node的embedding。所以，GCN是transductive的。（Transductive任务是指：训练阶段与测试阶段都基于同样的图结构）。而在实际应用中，由于新的item/user是源源不断的，所以我们的算法必须要支持新节点的加入；还有可能一个节点根本没有出现在训练集中，但是在测试集中出现（冷启动问题），所以我们的算法必须支持对未见节点的良好embedding。

而GraphSAGE是inductive的。inductive任务是指：训练阶段与测试阶段需要处理的graph不同。通常是训练阶段只是在子图（subgraph）上进行，测试阶段需要**处理未知的顶点**。

> ...leverage node **feature** information(e.g. text attributes) to efficiently generate node embeddings for previously unseen data. (也可以用此方法解决冷启动问题，即图中放入一个新的节点，如何快速得到它的embedding？可以利用它的属性特征，放入图中合适的位置。我们可以知道，GraphSAGE的每个节点初始化是用内容embedding来做的，所以就隐式的同时对**内容embedding**和**图关系embedding**进行了表征融合。)

**得到新节点的表示的难处：**

要想得到新节点的表示，需要让新的node或者subgraph去和已经优化好的node embedding去“对齐”。然而每个节点的表示都是受到其他节点的**影响**（牵一发而动全身），因此添加一个节点，意味着**许许多多与之相关的节点的表示都应该调整**。

**因此我们需要换一种思路：**

既然**新增的节点，一定会改变原有节点的表示**，那么我们**干嘛一定要得到每个节点的一个固定的表示呢？**我们何不直接**学习一种节点的表示方法**。这样不管graph怎么改变，都可以很容易地得到新的表示。具体的方法就是聚合邻居节点的表示。如何训练合适的聚合邻居节点的函数，就是GraphSAGE的关键。聚合一次邻居节点，我们就得到了“一阶关系”；聚合k次邻居节点，我们就得到了“k阶关系”。在测试的时候，面对新的节点，我们就可以用训练得到的聚合函数去聚合新节点的邻居，得到新节点的表征。



### 2. GraphSAGE

针对这种问题，GraphSAGE模型提出了一种算法框架，可以很方便地得到新node的表示。

#### 2.1 . Embedding generation（前向传播算法）

Embedding generation算法共**聚合**K次，总共有K个聚合函数(aggregator)，可以认为是K层，来聚合邻居节点的信息。假如用$h^k$来表示第k层每个节点的embedding，那么如何从$h^{k-1}$得到$h^k$呢？

- $h^{0}$就是初始的每个节点embedding。

- 对于每个节点v，都把它随机采样的若干**邻居**的**k-1**层的所有向量表示$\{h^{k-1}_u, u \in N(v)\}$、以及节点v**自己**的k-1层表示聚合成一个向量，然后把二者拼接起来，经过sigmoid激活函数的全连接层，就得到了第$k$层的表示$h^k$。这个聚合方法具体是怎么做的后面再详细介绍。

文中描述如下：

![img](https://pica.zhimg.com/80/v2-99e65d0ea27a2ba405dc81945189d628_1440w.jpeg)

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

  > By applying the max-pooling operator to each of the computed features, the model effectively captures different aspects of the neighborhood set.

#### 2.3 GraphSAGE的参数学习

GraphSAGE的参数就是**聚合函数AGGREGATE的参数**。为了学习这些参数，需要设计合适的损失函数。

对于**无监督学习**，设计的损失函数应该**让临近的节点的拥有相似的表示**，反之应该表示大不相同。所以损失函数是这样的：

![img](https://pic1.zhimg.com/80/v2-9c473f5e242f8db158854d4e5e036b9c_1440w.png)

其中，节点v是和节点u在一定长度的random walk上共现的节点，所以它们的点积要尽可能大；后面这项是采了Q个负样本，它们的点积要尽可能小。这个loss和skip-gram中的negative sampling如出一辙。

对于**有监督学习**，可以直接使用cross-entropy loss等常规损失函数。当然，上面的这个loss也可以作为一个**辅助loss**。



### 3. 和GCN的关系

原始GCN的方法，其实和GraphSAGE的Mean Pooling聚合方法是类似的，即每一层都聚合自己和自己邻居的归一化embedding表示。而GraphSAGE使用了其他capacity更大的聚合函数而已。

此外，GCN是一口气把整个图都丢进去训练，但是来了一个新的节点就不免又要把整个图重新训一次。而GraphSAGE则是在**增加了新的节点之后，来增量更新旧的节点，调整整张图的embedding表示**。只是生成新节点embedding的过程，实施起来相比于GCN更加灵活方便了。

----

参考：

https://zhuanlan.zhihu.com/p/74242097

