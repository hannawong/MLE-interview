

# MIND

这次介绍的论文是来自阿里的一篇文章Multi-Interest Network with Dynamic Routing for Recommendation at Tmall(下面简称MIND)，主要作用是学习**user embedding**，以用作**召回**。"the matching stage is responsible for retrieving thousands of candidate items that are relevant to user interests."

### 摘要

本篇文章是用于在**召回(matching)**阶段来建模用户兴趣(根据用户的历史行为来做兴趣的embedding)，**用来解决单个向量不能很好的表达用户多兴趣**的问题。



回忆一下以前对于用户embedding的工作：

- 协同过滤的隐向量(分解user-item交互矩阵)。缺点：sparsity problem, 计算资源耗费大;
- 深度网络中，用**单一向量**表示用户embedding（如之前所有交互过商品的mean-pooling）。缺点：bottleneck, 不能表示多样的用户兴趣。
- DIN模型使用attention来赋予不同历史商品不同的权重。缺点：计算极为**耗时**（因为粒度太细了，每次来一个item，user embedding都会发生变化），只能用于ranking(千级数据排序)，不能用于matching(亿级数据召回)，因为无法做到把item embedding离线存好、然后用ANN检索到和user相似的item。

本文提出的MIND可以**对一个用户输出多个embedding**,但是是比DIN更加粗粒度的**聚类**，每个embedding代表一类用户兴趣cluster。

### 3. METHOD

### 3.1 Problem Formalization

首先定义三元组：

![img](https://pic3.zhimg.com/v2-0d30a3b56ceb380f2159e89dad3fff4e_b.jpeg)

MIND的任务就是要根据用户的基础特征(gender,age...)以及用户历史行为来**计算她的embedding**。**注意这里的embedding是K个向量，它们代表着K个不同类的兴趣**。一个用户需要用向量组{v1,...vk}表示。

同时，对于亿级的商品，也对每个商品做embedding，为ei. $e_i = f_{item}(\mathbb{F}_i)$

召回阶段其实就是从这亿级商品中，选择top N(N=1000+)个和用户最为相关的商品。其相关性度量如下,即item embedding和K个user embedding的最大相似度（Maxsim）：

![img](https://pic1.zhimg.com/v2-b230d3e567563378b09316680f15dcc0_b.jpeg)

MIND模型网络结构：

![img](https://pic4.zhimg.com/v2-10c32a18fdeafb02a8323c25b16bf0f7_b.jpeg)





### 3.2 Embedding & Pooling Layer

Embedding & Pooling 层对应图中的最底层：

![img](https://pic1.zhimg.com/v2-8d62460afb840f395feac965043050f0_b.jpeg)

其中user profile的embedding做concat, 所有item的各个特征embedding做avg-pooling，得到一个item的embedding。

### 3.3 Multi-Interest Extract Layer

这部分介绍Multi-interest Extract Layer，这是用来给历史行为进行**聚类**，并且对每一个聚类生成一个embedding。这里的方法来源于dynamic routing for representation learning in capsule network, 因此先介绍**动态路由(dynamic routing)**。

![img](https://pic4.zhimg.com/v2-187f2e58d3e5055f73c80c2ffa847617_b.jpeg)

#### 3.3.1 动态路由

对于一层网络，第一层有m个节点，每个节点对应着一个长度为 ![N_l](https://www.zhihu.com/equation?tex=N_l) 的向量；第二层有n个节点，每个节点对应一个长度为 ![N_h](https://www.zhihu.com/equation?tex=N_h) 的向量。动态路由试图根据第一层的节点向量，通过迭代的方法，得到第二层的节点向量：

![img](https://pic3.zhimg.com/v2-5a34c48f9c95c12b905b2157e1a4c05a_b.jpeg)

实际上，每轮迭代都是把底层的向量$c_i$先经过MLP降维($N_l \rightarrow N_h$维)，然后通过加权求和得到上层的一个向量$c_j$。

这个过程主要分为三步：

- 计算权重 $w_{ij}$. 首先，我们先来计算$b_{ij}$, 即从底层向量$c_i$到上层向量$c_j$的权重。$b_{ij} = (\vec{c_j} S_{ij} \vec{c_i})$, 是一个标量。然后，对底层的m个向量对应的$b_{ij} $求softmax：![img](https://pic2.zhimg.com/80/v2-7f204e7b0813ee4a0ea8c69dbed8d84f_1440w.png)

- 第二步，进行加权平均。把底层的m个向量以权重$w_{ij}$进行加权求和，得到上层向量$z_j$.

![img](https://pic3.zhimg.com/80/v2-9edbad2a3d190d2bf47906dfc8ffbc6b_1440w.png)

- 第三步，把$z_j$进行归一化("squash")，得到$c_j $:

![img](https://pic2.zhimg.com/80/v2-907789160695714ac5e560e831861395_1440w.png)

迭代三次达到收敛。

**3.3.2 B2I (Behavior to Interest) 动态路由**

针对本文描述的情景，对原始的动态路由方法进行3方面改进：

（1）mapping矩阵S对于所有用户、所有节点都**【共享】**

![img](https://pic4.zhimg.com/v2-3e3bc3fb225e067a8a7564d8a04016c7_b.jpeg)

![img](https://pic3.zhimg.com/v2-3dbe93e936e920f2a149624718850c0e_b.jpeg)

（2）随机初始化，而不是初始化为全零

![img](https://pic4.zhimg.com/v2-e1917181535f512f9b3a26beae1a310f_b.jpeg)

（3）自适应调整用户兴趣类簇个数K。如果历史物品个数太少，应该将K调小。

![img](https://pic4.zhimg.com/v2-5492acf0e3c7ff60154b63deade401f3_b.jpeg)

### 3.4 Label-aware Attention Layer

label-aware attention layer对应图中的这个部分：

![img](https://pic3.zhimg.com/v2-1a215b747db243dcf642691f8642f322_b.jpg)

在训练阶段，使用label-aware attention来**计算target item和K个用户兴趣embedding的相关性**，赋予不同的权重。具体地，就是把target item作为Query，把K个兴趣聚类capsule作为Key和Value，求self-attention，这样我们就得到了user K个兴趣聚类更好的加权融合。这样，我们就得到了user端的embedding表示，这就是user塔的输出。item塔的输出就是target item embedding。

### 3.5 Training & Serving

1）训练阶段：

用户u购买/观看商品i的概率为：

![img](https://pic4.zhimg.com/v2-1f7e63f36b24a71780aa983be0358357_b.jpg)

由于分母要计算所有的商品，这实在是太多了。所以，我们采用类似word2vec中negative sampling的思想，抽取一些**负样本**，采用sampled softmax方法（和Youtube双塔一样）。同时使用Adam优化器。

2）serving阶段

训练之后，整个模型（除了Label-aware attention部分）都用于计算user embedding。我们把user的行为序列和profile都feed进模型，得到他的K个兴趣聚类capsule。然后，对于这每一个兴趣聚类embedding，都从item池中召回N个最接近的item，作为candidate。

当user有了新的行为，线上过来的时候他的兴趣聚类表征就会相应的发生变化，所以MIND可以进行实时的个性化召回。

### 4. Experiments

公开数据集：[Amazon Books](http://jmcauley.ucsd.edu/data/amazon)

工业数据集：TmallData, 200万用户行为

选择next-item prediction problem来进行召回，用hit-rate来判断召回算法的好坏：

![img](https://pic2.zhimg.com/v2-89b1668dfee63d27e6385be42ce7b6f5_b.jpg)

Baseline：

![img](https://pic2.zhimg.com/v2-0a93bf91b7744536013f2cdb86f620b9_b.jpg)

Online Experiment：

Baseline 选择item-based CF & Youtube DNN. 使用A/B test, 使用同样的ranking策略，最后比较CTR。

### 5. System Depolyment

用户打开app时，向Tmall Personality Platform提一个request。于是Tmall Personality Platform去找到该用户的bahaviors，经过User Interest Extractor提取出K个用户兴趣embedding。用此召回1000+个candidate items。经过ranking进行排序，取top-100展示。

![img](https://pic2.zhimg.com/v2-83eda61a1a572b7539fe126480e7af9d_b.jpeg)

Model Training Platform:

![img](https://pic4.zhimg.com/v2-31384d946491534f80efe45ee5460a57_b.jpeg)





【总结】

MIND可以看作是Youtube DNN的多兴趣优化，当K = 1时，MIND退化为Youtube DNN。

MIND的思想和ColBERT也十分类似，都是用多向量来表示query，然后求query和document表示的MaxSim，来得到query和document的相似度。毕竟，把一个query/document压缩成一个向量会造成信息的损失。