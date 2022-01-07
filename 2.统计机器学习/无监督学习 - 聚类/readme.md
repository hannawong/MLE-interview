# 无监督学习 - 聚类

无监督学习主要分为三类：**聚类**、**降维** (Dimensionality reduction)、**异常检测** (Outlier detection)。之前已经介绍了PCA降维，这篇文章主要来介绍聚类。聚类方法分为硬聚类（一个样本只属于一类）和软聚类（一个样本可以以不同概率属于多类，类似softmax），本文中我们只考虑硬聚类。

一个好的聚类算法，应当具有**大的类内相似度、小的类间相似度**。

### 0x01. K-Means聚类 – partitioning method

K-Means是知名度最高的一种聚类算法，代码非常容易理解和实现。

```python
[Algorithm 1] K-means:

随机初始化k个聚类质心点
迭代 t 步(或迭代到质心位置不再发生太大变化)
2.1 计算每个数据点到质心的距离来进行分类，它跟哪个聚类的质心更近，它就被分类到该聚类。
2.2 再重新计算每一聚类中所有向量的平均值，并确定出新的质心。
```

K-Means在本质上是一种EM算法(Expectation Maximization), 有关EM算法以后会详细介绍。

**K-means的优点：**

-  当数据的分布有明显的簇的划分时，K-means表现很好。(works well when the clusters are compact clouds that are well-separated from each other)

**K-means的缺点：** 

-  **k值**的选取不太确定。
-  一开始质心点的选取是**随机**的，不同的初始质心选择，会导致不同的结果。所以，K-means 一定能收敛，但不一定是最优解
-  对噪声和outlier非常敏感(因为直接以距离来度量)
- 只适用于**凸形状**聚类，不能用于非凸形状（凸形状是指当选择形状内任意两点，这两点的连线上所有点，也都在形状内）

K-Medians是与K-Means相关的另一种聚类算法，不同之处在于它使用簇的中值向量来重新计算质心点。和K-means不同，K-中值算法的聚类中心点一定是一个真实存在的点。该方法**对异常值不敏感**（因为使用中值），但在较大数据集上运行时速度会慢很多，因为每次计算中值向量，我们都要重新排序。

### 0x02. 层次聚类 -- hierarchical method

层次聚类实际上可以被分为两类：自上而下和自下而上。其中自下而上算法（Bottom-up algorithms）首先会将每个数据点视为单个聚类，然后连续合并（或聚合）成对的聚类，直到所有聚类合并成包含所有数据点的单个聚类。它也因此会被称为hierarchical agglomerative clustering。该算法的聚类可以被表示为一幅树状图，树根是最后收纳所有数据点的单个聚类，而树叶则是只包含一个样本的聚类。

```python
[Algorithm 2] 凝聚型层次聚类:

把每个数据点看作是一个聚类。
迭代，直到只有一个聚类簇：
2.1 找到距离最小的两个聚类簇c_i,c_j
2.2 将c_i,c_j合并为一个簇
```

三种衡量聚类簇相似性的标准：

- single linkage: 两个聚类簇中最相似的两个点的相似度
- complete linkage: 两个聚类簇中最不相似的两个点的相似度
- average linkage: 两个聚类簇中平均两两相似度

![img](https://pic1.zhimg.com/v2-3feeb1dc67417b5d71a9839093f38268_b.png)

层次聚类不要求我们指定聚类的数量，由于这是一个构建树的过程，我们甚至可以选择那种聚类看起来更合适。它具有 ![O(n^3)](https://www.zhihu.com/equation?tex=O(n%5E3))的时间复杂度。

### 0x03. DBSCAN – density-based method

#### 1. 简介

DBSCAN（Density-Based Spatial Clustering of Applications with Noise，具**有噪声的基于密度的聚类方**法）是一种基于密度的空间聚类算法。 该算法将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇。

它将簇定义为密度相连的点的最大集合。通过将紧密相连的样本划为一类，这样就得到了一个聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，则我们就得到了最终的所有聚类类别结果。

我们知道k-means聚类算法只能处理凸(球形)的簇，也就是一个聚成实心的团 (cloud), 这是因为算法本身计算平均距离的局限。但往往现实中还会有各种形状，比如下面这张图，传统的聚类算法显然无法解决。

![img](https://pic2.zhimg.com/v2-5021447bcd66976bee25c16d3d78c679_b.png)



#### 2. 重要概念

假设样本集 ![D = ( x_1 , x_2 , . . . , x_m )](https://www.zhihu.com/equation?tex=D%20%3D%20(%20x_1%20%2C%20x_2%20%2C%20.%20.%20.%20%2C%20x_m%20)), 则DBSCAN具体的密度描述定义如下:

(1) ![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)**邻域：**对于一个样本 ![x_j](https://www.zhihu.com/equation?tex=x_j), 其![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon) 邻域包含样本集D中所有与 ![x_j](https://www.zhihu.com/equation?tex=x_j) 距离不大于 ![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon) 的样本点集合，这个集合的元素个数记作 ![|N\epsilon(x_j)|](https://www.zhihu.com/equation?tex=%7CN%5Cepsilon(x_j)%7C) .

(2) **核心对象**：对于任一样本 ![x_j](https://www.zhihu.com/equation?tex=x_j) 如果其ϵ-邻域对应的ϵ邻域至少包含MinPts个样本，则 ![x_j](https://www.zhihu.com/equation?tex=x_j) 是核心对象。

(3) **直接密度可达**：如果 ![x_i](https://www.zhihu.com/equation?tex=x_i) 位于 ![x_j](https://www.zhihu.com/equation?tex=x_j) 的ϵ-邻域中，且 ![x_j](https://www.zhihu.com/equation?tex=x_j) 是核心对象，则称 ![x_i](https://www.zhihu.com/equation?tex=x_i) 由 ![x_j](https://www.zhihu.com/equation?tex=x_j) 直接密度可达。注意反之不一定成立，即此时不能说 ![x_j](https://www.zhihu.com/equation?tex=x_j) 由 ![x_i](https://www.zhihu.com/equation?tex=x_i) 密度直达, 除非 ![x_i](https://www.zhihu.com/equation?tex=x_i) 也是核心对象。

(4) **密度可达：**对于 ![x_i](https://www.zhihu.com/equation?tex=x_i) 和 ![x_j](https://www.zhihu.com/equation?tex=x_j) ,如果存在样本序列 ![\{p_1,p_2,...,p_T\}](https://www.zhihu.com/equation?tex=%5C%7Bp_1%2Cp_2%2C...%2Cp_T%5C%7D), 满足![p_1 = x_i ,p_T = x_j](https://www.zhihu.com/equation?tex=p_1%20%3D%20x_i%20%2Cp_T%20%3D%20x_j), 其中对于所有的 ![p_{t+1}](https://www.zhihu.com/equation?tex=p_%7Bt%2B1%7D) ,它都可以由 $p_{t}$ 直接密度可达。也就是说，所有的 $$\{p_1,p_2,...,p_{T-1}\}$$ 都是核心对象。称 ![x_j](https://www.zhihu.com/equation?tex=x_j) 由 ![x_i](https://www.zhihu.com/equation?tex=x_i) 密度可达。也就是说，密度可达就是直接密度可达的传递。注意密度可达也不满足对称性，这个可以由直接密度可达的不对称性得出。

(5) **密度相连：**对于 ![x_i](https://www.zhihu.com/equation?tex=x_i) 和 ![x_j](https://www.zhihu.com/equation?tex=x_j) ,如果存在核心对象 ![x_k](https://www.zhihu.com/equation?tex=x_k) ,使 ![x_i](https://www.zhihu.com/equation?tex=x_i) 和 ![x_j](https://www.zhihu.com/equation?tex=x_j) 均由 ![x_k](https://www.zhihu.com/equation?tex=x_k) 密度可达，则称![x_i](https://www.zhihu.com/equation?tex=x_i) 和 ![x_j](https://www.zhihu.com/equation?tex=x_j)密度相连。注意密度相连关系是满足对称性的。



从下图可以很容易看出理解上述定义，图中MinPts=5，红色的点都是核心对象，因为其ϵ-邻域至少有5个样本。黑色的样本是非核心对象。所有核心对象**直接密度可达**的样本在以红色核心对象为中心的超球体内，如果不在超球体内，则不能直接密度可达。图中用绿色箭头连起来的核心对象组成了**密度可达**的样本序列。在这些密度可达的样本序列的ϵ-邻域内所有的样本相互都是**密度相连**的。

所以，下图中共有两个密度相连的簇（左边和右边），簇中的每个样本都是密度相连的。

![img](https://pic1.zhimg.com/v2-bf304fca90a313772765ec4abd9d5adc_b.png)

定义：

- 边界点(border point)在ϵ邻域内的点小于MinPts个，但它落在一个核心点的ϵ邻域内。

- 噪声点(noise point)不在任何一个核心对象在周围. 所以，DBSCAN是可以解决噪声点的。

![img](https://pic4.zhimg.com/v2-a0b49d6fbf78a1d08c9ff25005d66b8f_b.png)

​                                       右图中，绿色的点是核心对象，蓝色的点是边界点，红色的点是噪声。

#### 3. DBSCAN算法

```
[Algorithm 3]: DBSCAN

随机选一点p
根据参数Minpts和ϵ找到p点密度可达的一些点
2.1 如果p是核心对象，则一个聚类簇已经形成了；
2.2 如果p不是核心对象，没有点是从p开始密度可达的，于是跳过它访问下一个对象。
持续迭代，直到所有的点都被访问过。
```

时间复杂度 ![ O(N^2)](https://www.zhihu.com/equation?tex=%20O(N%5E2))  , 使用kd树优化可以达到 ![O(NlogN)](https://www.zhihu.com/equation?tex=O(NlogN)) .

#### 4. 小结

我们什么时候需要用DBSCAN来聚类呢？一般来说，如果数据集是**稠密**的，并且数据集是**非凸**的，那么用DBSCAN会比K-Means聚类效果好很多。如果数据集不是稠密的，则不推荐用DBSCAN来聚类。

DBSCAN的优点有：

- 可以对**任意形状的稠密数据集**进行聚类。相对的，K-Means之类的聚类算法一般只适用于凸数据集。
- 可以在聚类的同时发现异常点，对数据集中的异常点不敏感。("With **noise**")
- 聚类结果不受初始值的影响。相对的，K-Means之类的聚类算法**初始值**对聚类结果有很大影响。

DBSCAN的缺点有：

- 如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。
- 如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。
- **调参**相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值ϵ，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有很大影响。


  