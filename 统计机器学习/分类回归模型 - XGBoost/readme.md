# XGBoost

下图描述了树模型的演变：

![img](https://pic2.zhimg.com/v2-0243a69499d994271844d252df1723a1_b.png)



------

### 1. 简介

   XGBoost 算法最开始是作为华盛顿大学的一个研究项目开发的。 Tianqi Chen 和 Carlos Guestrin 在 2016 年的 SIGKDD 会议上发表了XGBoost的[论文](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)，引起强烈反响。自推出以来，该算法不仅赢得了众多 Kaggle 比赛，而且还成为行业应用的推动力。XGBoost 开源项目在 [Github](https://github.com/dmlc/xgboost/) 上有 500+名contributor和 5000+次commit。

   XGBoost拥有明显的优势：

- 应用广泛。可用于解决回归、分类、排序和用户定义的预测问题。
- 可移植性。在 Windows、Linux 和 OS X 上流畅运行。
- 支持所有主要的编程语言。包括 C++、Python、R、Java、Scala 和 Julia。
- 云集成。支持 AWS、Azure 和 Yarn 集群，与 Flink、Spark 等**分布式**生态系统配合良好，使得它可以很好地解决工业界大规模数据的问题。

   本文将介绍XGboost算法的推导和特性。由于笔者数学很差，所以尽量用不那么mathy的方式来讲解。

### 2. XGBoost的原理推导

​    XGBoost和GBDT两者都是boosting方法，除了工程实现、解决问题上的一些差异外，最大的不同就是**损失函数**的定义。因此，本文我们从目标函数开始探究XGBoost的基本原理。

​    XGBoost的损失函数表示为：

![L = \sum_{i=1}^nl(\hat{y_i},y_i)+\sum_k\Omega(f_k)](https://www.zhihu.com/equation?tex=L%20%3D%20%5Csum_%7Bi%3D1%7D%5Enl(%5Chat%7By_i%7D%2Cy_i)%2B%5Csum_k%5COmega(f_k))

​    其中，n为样本数量；k是树的个数，![\Omega(f_k)](https://www.zhihu.com/equation?tex=%5COmega(f_k))是第k棵树的**复杂度**， ![\sum_k\Omega(f_k)](https://www.zhihu.com/equation?tex=%5Csum_k%5COmega(f_k)) 就是将k棵树的复杂度求和。

​    我们知道模型的预测精度由模型的**偏差和方差**共同决定，损失函数的前半部分代表了模型的偏差；想要方差小则需要在目标函数中添加正则项，用于防止过拟合。所以目标函数的后半部分是抑制**模型复杂度的正则项** ![\Omega(f_k)](https://www.zhihu.com/equation?tex=%5COmega(f_k))，这也是XGBoost和GBDT的差异所在。

​    树的复杂度计算公式如下：

​                                                                   ![\Omega(f) = \gamma T+\frac{1}{2} \lambda\sum_{j=1}^Tw_j^2](https://www.zhihu.com/equation?tex=%5COmega(f)%20%3D%20%5Cgamma%20T%2B%5Cfrac%7B1%7D%7B2%7D%20%5Clambda%5Csum_%7Bj%3D1%7D%5ETw_j%5E2)  


​    其中， T 为树的叶子节点个数，![w_j](https://www.zhihu.com/equation?tex=w_j)为第 j 个节点的权重。叶子节点越少模型越简单，此外叶子节点也不应该含有过高的权重.

**2.2 Gradient Tree Boosting**

**2.2.1 目标函数推导**

​    由于XGBoost是boosting族中的算法，所以遵从前向分步加法，以第 t 步的模型为例，模型对第 i 个样本 $x_i$ 的预测值为：

![img](https://pic3.zhimg.com/v2-81b214bdf0882bc2ea094868d8fa71ba_b.jpeg)

​                                                                            (前向分步加法)

将此公式代入原先的损失函数中：

![img](https://pic4.zhimg.com/v2-e1a29a11058fef6178039bed0b13a3eb_b.jpeg)

注意上式中，**只有一个变量**，那就是第 t 棵树 $f_t$ ，其余都是通过已知量可以计算出来的。细心的同学可能会问，上式中的第二行到第三行是如何得到的呢？这里我们将正则化项进行拆分，由于前 t-1 棵树的结构已经确定，因此前 t-1 棵树的复杂度之和可以用一个常量const表示。

回忆泰勒展开：

![img](https://pic4.zhimg.com/v2-8942d6265a55fd0f43f793c04446a1bf_b.jpeg)

对目标损失函数用泰勒展开，得到：

![img](https://pic3.zhimg.com/v2-35b23ffb80cadac1b18483b737632382_b.jpeg)

​    其中 $g_i$ 是损失函数  ![l(y_i,\hat{y_i}^{(t-1)})](https://www.zhihu.com/equation?tex=l(y_i%2C%5Chat%7By_i%7D%5E%7B(t-1)%7D))  对   ![\hat{y_i}^{(t-1)}](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D%5E%7B(t-1)%7D)  的一阶导。读过我前面写的GBDT文章的同学应该发现了，  ![g_i](https://www.zhihu.com/equation?tex=g_i)   就是在梯度提升树中第t棵树要拟合的那个值（如果是平方误差损失的回归问题，就是残差。）

​    但是，XGBoost 还引入了二阶导 $h_i$ 。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。

​    j将 ![\Omega(f_t)](https://www.zhihu.com/equation?tex=%5COmega(f_t)) 的公式代入上式，经过一系列推导（省略），得到**损失函数的最小值min(Obj)**：

![img](https://pic4.zhimg.com/v2-102a5435bd0fcc53ee3ccc76b726f19f_b.jpeg)

​    其中，T为这棵树的**叶节点**个数，G_j 为叶子节点 j 所含**所有样本的一阶导数** g_i 之和；H_j 为节点 j 所含所有样本的二阶导数 h_i 之和。



**2.2.2 分裂算法**

**算法1：Exact Greedy Algorithm**

XGBoost使用了和CART回归树一样的想法，利用贪婪算法，将所有特征升序排序，然后遍历所有特征的所有特征划分点，不同的是使用的目标函数不一样。具体做法就是求分裂后的目标函数值比分裂前的目标函数的增益。

![img](https://pic4.zhimg.com/v2-2929c1dfc26a032922489bc27f86851f_b.jpeg)

**算法2：近似算法**

   贪心算法可以得到最优解，但当数据量太大时则无法读入内存进行计算，近似算法则大大降低了计算量，给出了近似最优解。

   对于每个特征，只考察分位点可以减少计算复杂度。近似算法简单来说，就是对每个特征 k​ 都确定 l 个候选切分点 ![S_k = \{S_{k1}, S_{k2},...S_{kl}\}](https://www.zhihu.com/equation?tex=S_k%20%3D%20%5C%7BS_%7Bk1%7D%2C%20S_%7Bk2%7D%2C...S_%7Bkl%7D%5C%7D) ，然后根据这些候选切分点把相应的样本放入对应的桶中，对每个桶的g_i,h_i 进行累加。最后在候选切分点集合上贪心查找。例如：

![img](https://pic1.zhimg.com/v2-c04f7b4e428db4e1a8f57d1bc01e6cec_b.png)

![img](https://pic4.zhimg.com/v2-a44ba14b045f203345d180d46e9d7b1f_b.jpeg)

**2.3 缩减和列抽样(shrinkage and column sampling)**

​    除了增加![\sum_k\Omega(f_k)](https://www.zhihu.com/equation?tex=%5Csum_k%5COmega(f_k))正则项之外，作者还选用了两个正则化方法。

​    一是shrinkage。XGBoost 在进行完一次迭代后，会将新加进来的树的叶子节点权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。传统GBDT的实现也有**学习速率**；步长越小，越有可能找到更精确的最佳值，更多的空间被留给了后面建立的树，但迭代速度会比较缓慢。

> Shrinkage scales newly added weights by a factor η after each step of tree boosting. Similar to a learning rate in stochastic optimization, shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model. 

![img](https://pic2.zhimg.com/v2-cbfa8f6583104d1055d7493182ed9d0d_b.png)

​    二是列抽样。XGBoost借鉴了**随机森林**的做法，每次分裂的时候随机选择m个特征计算增益，而不是全部特征都考虑。不仅能降低过拟合，还能减少计算。这也是XGBoost异于传统GBDT的一个特性。

**2.4 解决数据缺失问题(Sparsity-aware Split Finding)**

​    在现实应用中，输入x很有可能是稀疏的。有很多因素导致了数据的稀疏性：

- 数据缺失
- 数据中频繁出现0
- one-hot encoding导致的数据稀疏

​    为了解决数据的稀疏性，我们为每个节点增加了一个default direction, 当一个特征的数据缺失时，就被自动归到这个方向:

![img](https://pic4.zhimg.com/v2-04d9a2ce481a62aceea2de4b55f695fb_b.png)

​    在逻辑实现上，会分别处理将该特征值为missing/0的样本分配到左叶子结点和右叶子结点的两种情形，计算增益后选择增益大的方向进行分裂即可。如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子树。

![img](https://pic3.zhimg.com/v2-67a2b34c5a44bd1c68a3e982b947de0a_b.png)

**2.5 并行计算**

​    Boosting不是一种串行的结构吗?怎么并行的？

​    注意XGBoost的并行不是树粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）。而XGBoost在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么**各个特征的增益计算就可以开多线程进行**。

### 3. 实验结果

![img](https://pic3.zhimg.com/v2-945782246f7f30c8b2daf98f037122ca_b.png)



----

参考资料：

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63dtowardsdatascience.com


