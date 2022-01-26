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


### 2. XGBoost的原理推导

​    XGBoost和GBDT两者都是boosting方法，除了工程实现、解决问题上的一些差异外，最大的不同就是**损失函数**的定义。因此，本文我们从目标函数开始探究XGBoost的基本原理。

​    XGBoost的损失函数表示为：

![L = \sum_{i=1}^nl(\hat{y_i},y_i)+\sum_k\Omega(f_k)](https://www.zhihu.com/equation?tex=L%20%3D%20%5Csum_%7Bi%3D1%7D%5Enl(%5Chat%7By_i%7D%2Cy_i)%2B%5Csum_k%5COmega(f_k))

​    其中，n为样本数量；k是树的个数，![\Omega(f_k)](https://www.zhihu.com/equation?tex=%5COmega(f_k))是第k棵树的**复杂度**， ![\sum_k\Omega(f_k)](https://www.zhihu.com/equation?tex=%5Csum_k%5COmega(f_k)) 就是将k棵树的复杂度求和。

​    我们知道模型的预测精度由模型的**偏差和方差**共同决定，损失函数的前半部分代表了模型的偏差；想要方差小则需要在目标函数中添加正则项，用于防止过拟合。所以目标函数的后半部分是抑制**模型复杂度的正则项** ![\Omega(f_k)](https://www.zhihu.com/equation?tex=%5COmega(f_k))，这也是XGBoost和GBDT的差异所在。

​    一棵树的复杂度计算公式如下：

​                                                                   ![\Omega(f) = \gamma T+\frac{1}{2} \lambda\sum_{j=1}^Tw_j^2](https://www.zhihu.com/equation?tex=%5COmega(f)%20%3D%20%5Cgamma%20T%2B%5Cfrac%7B1%7D%7B2%7D%20%5Clambda%5Csum_%7Bj%3D1%7D%5ETw_j%5E2)  


​    其中， T 为树的叶子节点个数，![w_j](https://www.zhihu.com/equation?tex=w_j)为第 j 个节点的权重。叶子节点越少模型越简单，此外叶子节点也不应该含有过高的权重.

**2.2 Gradient Tree Boosting**

**2.2.1 目标函数推导**

​    由于XGBoost是boosting族中的算法，所以遵从前向分步加法，以第 t 步的模型为例，模型对第 i 个样本 $x_i$ 的预测值为：

![img](https://pic3.zhimg.com/v2-81b214bdf0882bc2ea094868d8fa71ba_b.jpeg)

​                                                                            (前向分步加法)

将此公式代入原先的损失函数中，即我们要在原先那些树的基础上再增加一棵新的树：

![img](https://pic4.zhimg.com/v2-e1a29a11058fef6178039bed0b13a3eb_b.jpeg)

注意上式中，**只有一个变量**，那就是第 t 棵树 f_t ，其余都是通过已知量可以计算出来的。细心的同学可能会问，上式中的第二行到第三行是如何得到的呢？这里我们将正则化项进行拆分，由于前 t-1 棵树的结构已经确定，因此前 t-1 棵树的复杂度之和可以用一个常量const表示。

回忆泰勒展开：

![img](https://pic4.zhimg.com/v2-8942d6265a55fd0f43f793c04446a1bf_b.jpeg)

对目标损失函数用泰勒展开，得到：

![img](https://pic3.zhimg.com/v2-35b23ffb80cadac1b18483b737632382_b.jpeg)

​    其中 $g_i$ 是损失函数  ![l(y_i,\hat{y_i}^{(t-1)})](https://www.zhihu.com/equation?tex=l(y_i%2C%5Chat%7By_i%7D%5E%7B(t-1)%7D))  对   ![\hat{y_i}^{(t-1)}](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D%5E%7B(t-1)%7D)  的一阶导，这个东西就叫“**样例i的梯度**”。读过我前面写的GBDT文章的同学应该发现了，-  ![g_i](https://www.zhihu.com/equation?tex=g_i)   就是在梯度提升树中第t棵树要拟合的那个值（如果是平方误差损失的回归问题，就是残差。）

​    但是，XGBoost 还引入了二阶导 $h_i$ 。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。

​    j将 ![\Omega(f_t)](https://www.zhihu.com/equation?tex=%5COmega(f_t)) 的公式代入上式，经过一系列推导（省略），得到**损失函数的最小值min(Obj)**：

![img](https://pic4.zhimg.com/v2-102a5435bd0fcc53ee3ccc76b726f19f_b.jpeg)

​    其中，T为这棵树的**叶节点**个数，G_j 为叶子节点 j 所含**所有样本的一阶导数** g_i 之和；H_j 为节点 j 所含所有样本的二阶导数 h_i 之和。因此，**第t棵树就是要让上面的这个值最小**。这个值就像决策树中的impurity一样，需要用greedy方法来最小化之。



**2.2.2 分裂算法**

**算法1：Exact Greedy Algorithm**

XGBoost使用了和CART回归树一样的想法，利用贪婪算法，将所有特征升序排序，然后遍历所有特征的所有特征划分点，不同的是使用的目标函数不一样。具体做法就是求分裂后的目标函数值比分裂前的目标函数的增益。

![img](https://pic4.zhimg.com/v2-2929c1dfc26a032922489bc27f86851f_b.jpeg)

**算法2：近似算法**

   贪心算法可以得到最优解，但当数据量太大时则无法读入内存进行计算，近似算法则大大降低了计算量，给出了近似最优解。

   对于每个特征，**只考察分位点以减少计算复杂度**。近似算法简单来说，就是对每个特征 k 都确定 l 个候选切分点 ![S_k = \{S_{k1}, S_{k2},...S_{kl}\}](https://www.zhihu.com/equation?tex=S_k%20%3D%20%5C%7BS_%7Bk1%7D%2C%20S_%7Bk2%7D%2C...S_%7Bkl%7D%5C%7D) ，然后根据这些候选切分点把相应的样本放入对应的桶中，对每个桶的g_i,h_i 进行累加。最后在候选切分点集合上贪心查找。

在提出候选切分点时有两种策略：

- Global：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
- Local：每次分裂前将重新提出候选切分点。

![img](https://pic1.zhimg.com/80/v2-1fe2882f8ef3b0a80068c57905ceaba0_1440w.jpg)



例如：

![img](https://pic1.zhimg.com/v2-c04f7b4e428db4e1a8f57d1bc01e6cec_b.png)

![img](https://pic4.zhimg.com/v2-a44ba14b045f203345d180d46e9d7b1f_b.jpeg)

**2.2.3 加权分位数缩略图**

实际上，XGBoost不是简单地按照样本个数进行分位，而是以二阶导数值 ![[公式]](https://www.zhihu.com/equation?tex=h_i+) 作为**样本的权重**进行划分。为了处理带权重的候选切分点的选取，作者提出了Weighted Quantile Sketch算法。加权分位数略图侯选点的选取方式，如下：

![img](https://pic2.zhimg.com/80/v2-5549ade8baaf0587fbbe1834dc564771_1440w.jpg)

**那么为什么要用二阶梯度 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 进行样本加权？**

我们知道模型的目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=+Obj%5E%7B%28t%29%7D+%5Csimeq+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C)

我们把目标函数配方整理成以下形式，便可以看出 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 有对 loss 加权的作用。

![img](https://pic1.zhimg.com/80/v2-bf5ad02673f8d55133e6429b33c815bb_1440w.jpeg)

这个式子看上去就是一个weighted squared loss, 以h_i为weight，以g_i/h_i为label。



**2.3 缩减和列抽样(shrinkage and column sampling)**

​    除了增加![\sum_k\Omega(f_k)](https://www.zhihu.com/equation?tex=%5Csum_k%5COmega(f_k))正则项之外，作者还选用了**两个正则化方法**。

​    一是shrinkage。XGBoost 在进行完一次迭代后，会将新加进来的树的叶子节点权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。传统GBDT的实现也有**学习速率**；步长越小，越有可能找到更精确的最佳值，更多的空间被留给了后面建立的树，但迭代速度会比较缓慢。

> Shrinkage scales newly added weights by a factor η after each step of tree boosting. Similar to a learning rate in stochastic optimization, shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model. 

![img](https://pic2.zhimg.com/v2-cbfa8f6583104d1055d7493182ed9d0d_b.png)

​    二是列抽样。XGBoost借鉴了**随机森林**的做法，每次分裂的时候**随机选择m个特征**计算增益，而不是全部特征都考虑。不仅能降低过拟合，还能减少计算。这也是XGBoost异于传统GBDT的一个特性。



**2.4 解决数据缺失问题(Sparsity-aware Split Finding)**

​    在现实应用中，输入x很有可能是稀疏的。有很多因素导致了数据的稀疏性：

- 数据缺失
- 数据中频繁出现0
- one-hot encoding导致的数据稀疏

​    为了解决数据的稀疏性，我们为每个节点增加了一个default direction, 当一个特征的数据缺失时，就被自动归到这个方向:

![img](https://pic4.zhimg.com/v2-04d9a2ce481a62aceea2de4b55f695fb_b.png)

​    在逻辑实现上，会分别处理将该特征值为missing的样本分配到左叶子结点和右叶子结点的两种情形，计算增益后选择增益大的方向进行分裂即可。如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子树。

![img](https://pic3.zhimg.com/v2-67a2b34c5a44bd1c68a3e982b947de0a_b.png)



### 3. Xgboost工程实现

#### 3.1 **列块并行学习** （Column Block for Parallel Learning）

在树生成过程中，最耗时的一个步骤就是在每次寻找**最佳分裂点时都需要对特征的值进行排序**。而 XGBoost **在训练之前会根据特征对数据进行排序**，然后保存到块结构中，并在每个块结构中都采用了稀疏矩阵存储格式（Compressed Sparse Columns Format，CSC）进行存储，后面的训练过程中会重复地使用块结构，可以大大减小计算量。

作者提出通过**按特征进行分块并排序**，每个特征占一块。在块里面保存排序后的特征的值(Feature Values，红色方框)及**对应样本的一阶、二阶导数值**（黄色圆圈）。具体方式如图：

![img](https://pic2.zhimg.com/80/v2-3a93e4d9940cf6e2e9fd89dfa38dc62d_1440w.jpg)



通过顺序访问排序后的块遍历样本特征的特征值，方便进行切分点的查找。

分块存储后多个特征之间互不干涉，可以使用**多线程同时对不同的特征进行切分点查找**，即特征的并行化处理。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是 XGBoost 能够实现分布式或者多线程计算的原因。

【面试题：Xgboost是怎么并行化的？Boosting不是一种串行的结构吗，怎么并行的？】

答：注意XGBoost的并行不是树粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）。而XGBoost在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么**各个特征的增益计算就可以开多线程进行**。



#### 3.2 缓存访问 (Cache-aware access)

列块并行学习的设计可以减少节点分裂时的计算量，在顺序访问特征值时，访问的是一块连续的内存空间，但通过特征值持有的**索引访问样本获取一阶、二阶导数**时，这个访问操作访问的内存空间并不连续，这样可能造成cpu缓存命中率低，影响算法效率。

为了解决缓存命中率低的问题，XGBoost 提出了缓存访问算法：为每个线程**分配一个连续的缓存区**，将需要的梯度信息存放在缓冲区中，这样就实现了非连续空间到连续空间的转换，提高了算法效率。此外适当调整块大小，也可以有助于缓存优化。



#### **3.3 “核外”块计算** (Blocks for Out-of-core Computation)

当数据量非常大时，我们不能把所有的数据都加载到内存中。那么就必须将一部分需要加载进内存的数据先存放在**硬盘**中，当需要时再加载进内存。这样操作具有很明显的瓶颈，即硬盘的IO操作速度远远低于内存的处理速度，肯定会存在大量等待硬盘IO操作的情况。针对这个问题作者提出了“核外”计算的优化方法。具体操作为，将数据集分成多个块存放在硬盘中，**使用一个独立的线程专门从硬盘读取数据，加载到内存中**，这样算法在内存中处理数据就可以和从硬盘读取数据同时进行。此外，XGBoost 还用了两种方法来降低硬盘读写的开销：

- **块压缩**（**Block Compression**）。论文使用的是按列进行压缩，读取的时候用另外的线程解压。对于行索引，只保存第一个索引值，然后用16位的整数保存与该block第一个索引的差值。作者通过测试在block设置为 ![[公式]](https://www.zhihu.com/equation?tex=2%5E%7B16%7D) 个样本大小时，压缩比率几乎达到26% ![[公式]](https://www.zhihu.com/equation?tex=%5Csim) 29%。
- **块分区**（**Block Sharding** ）。块分区是将特征block分区存放在不同的硬盘上，以此来增加硬盘IO的吞吐量。



## **4. XGBoost的优缺点**

**4.1 优点**

- **精度更高：**GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
- **灵活性更强：**GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导；
- **正则化：**XGBoost 在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的 L2 范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合，这也是XGBoost优于传统GBDT的一个特性。
- **Shrinkage（缩减）：**相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。传统GBDT的实现也有学习速率；
- **列抽样：**XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。这也是XGBoost异于传统GBDT的一个特性；
- **缺失值处理：**对于特征的值有缺失的样本，XGBoost 采用的稀疏感知算法可以自动学习出它的分裂方向；
- **XGBoost工具支持并行：**boosting不是一种串行的结构吗?怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。
- **可并行的近似算法：**树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以XGBoost还提出了一种可并行的近似算法，用于高效地生成候选的分割点。

**4.2 缺点**

- 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
- 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。

#### 5. 参数

在上一部分中，XGBoost模型的参数都使用了模型的默认参数，但默认参数并不是最好的。要想让XGBoost表现的更好，需要对XGBoost模型进行参数微调。下图展示的是分类模型需要调节的参数，回归模型需要调节的参数与此类似。

![img](https://pic4.zhimg.com/80/v2-323f135041adb7e7763af1544305a5c7_1440w.jpg)







## 6. 关于XGBoost若干问题的思考

**6.1 XGBoost与GBDT的联系和区别有哪些？**

（1）GBDT是机器学习算法，XGBoost是该算法的工程实现。

（2）**正则项：**在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。

（3）**导数信息：**GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。

（4）**基分类器：**传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类器，比如线性分类器。

（5）**子采样：**传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样。

（6）**缺失值处理：**传统GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺失值的处理策略。

（7）**并行化**：传统GBDT没有进行并行化设计，注意不是tree维度的并行，而是特征维度的并行。XGBoost预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度。

**6.2 为什么XGBoost泰勒二阶展开后效果就比较好呢？**

二阶信息本身就能让梯度收敛更快更准确。这一点在优化算法里的**牛顿法**中已经证实。可以简单认为一阶导指引梯度方向，二阶导指引梯度方向如何变化。简单来说，相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数。

**6.3 XGBoost对缺失值是怎么处理的？**

在普通的GBDT策略中，对于缺失值的方法是先手动对缺失值进行填充，然后当做有值的特征进行处理，但是这样人工填充不一定准确，而且没有什么理论依据。而XGBoost采取的策略是先不处理那些值缺失的样本，采用那些有值的样本搞出分裂点，在遍历每个有值特征的时候，尝试将缺失样本划入左子树和右子树，选择使损失最优的值作为分裂点。

**6.4 XGBoost为什么可以并行训练？**

（1）XGBoost的并行，并不是说每棵树可以并行训练，XGBoost本质上仍然采用boosting思想，每棵树训练前需要等前面的树训练完成才能开始训练。

（2）XGBoost的并行，指的是特征维度的并行：在训练之前，每个特征按特征值对样本进行预排序，并存储为Block结构，在后面查找特征分割点时可以重复使用，而且特征已经被存储为一个个block结构，那么在寻找每个特征的最佳分割点时，可以利用多线程对每个block并行计算。





参考资料：

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63dtowardsdatascience.com

https://zhuanlan.zhihu.com/p/83901304


