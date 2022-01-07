# 决策树

决策树可以用来解决分类(classification)和回归(regression)问题。由于其有出色的**可解释性**(可以直接得到feature importance)及**对缺失值不敏感**的特性，广泛用于搜索/广告及风控场景。比如在搜索/广告场景可以通过决策树来进行组合特征抽取(如GBDT+LR来得到特征交互)，风控场景可以评估是否能借贷给目标人。

下面将介绍关于决策树的一些知识点。

## 0x01. ID3 by Quinlan

对于分类问题，ID3的做法是：

- 自顶向下，贪心搜索 **(top-down, greedy search)**。因为没有回溯(back-tracking)，所以会导致**局部最优**、而不是全局最优
- 递归地完成如下步骤：
  - 对于信息增益最大的那个特征A作为下一步的分裂特征
  - 对于特征A的每一个取值 ![A_{v_i}](https://www.zhihu.com/equation?tex=A_%7Bv_i%7D),都分配一个节点，并将样例点分配给相应的节点
- 如果所有样例点都被**完美地分类**（不太可能）/**符合终止条件**，则RETURN。否则，继续递归处理该节点。

#### 1. 信息增益

我们需要按照什么标准构造一颗树呢？由于决策树是自顶向下递归方式，每一步使用贪心算法采用当前状态下最优选择。所以我们构造决策树的关键就是选择一个好的分裂标准。

ID3使用信息增益作为特征选择的度量。那么什么是信息增益呢？要明白信息增益，首先需要知道信息熵。

##### 1.1 衡量系统的混乱程度 -- 熵

​    假设随机变量X的取值为$\{x_{1},x_{2}...x_{n}\}$，对应出现的概率为$\{p_{1},p_{2}...p_{n}\}$。X的信息熵用来衡量系统的混乱程度，表示为：

​                                                                             ![H(X)=-\sum_{i=1}^{n}{p_{i}log_{2}p_{i}}](https://www.zhihu.com/equation?tex=H(X)%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bp_%7Bi%7Dlog_%7B2%7Dp_%7Bi%7D%7D)

可见，均匀分布有着最大的信息熵。

**基尼系数**（GINI impurity）也可以用来衡量系统的混乱程度。

![G(X) = 1-\sum_{i=1}^np_i^2](https://www.zhihu.com/equation?tex=G(X)%20%3D%201-%5Csum_%7Bi%3D1%7D%5Enp_i%5E2)

##### 1.2 信息增益

信息增益就是：得知特征A的信息而使得样本集合不确定性**（熵）减少**的程度。

> we measure the change of impurity ΔI(N) —Information Gain (IG), to estimate the expected reduction in entropy due to sorting on A 

![img](https://pic1.zhimg.com/v2-73168414859fd226d49e392617ad3998_b.png)

示例：

![img](https://pic3.zhimg.com/v2-3428face77eca7a34d073349e76f8bca_b.png)

现有两个特征 ![A_1](https://www.zhihu.com/equation?tex=A_1)和 ![A_2](https://www.zhihu.com/equation?tex=A_2) ,我们要判断哪个特征是信息增益大的那个。原始的信息熵为0.993.

##### 1.2 终止条件

上文提到，节点终止分裂的条件是：

> 如果所有样例点都被**完美**地分类/符合**终止条件**，则RETURN。

那么，什么叫做“完美地分类”呢？“完美分类”的定义是：

- Condition 1: if all the data in the current subset has the same output class, then stop.(pure enough)
- Condition 2: if all the data in the current subset has the same input value, then stop.(you have to assign a value randomly).

当然，这两个标准太过严苛。在实际应用中，我们会采取更宽松的停止条件，见下一节"树的过拟合和剪枝".

##### 1.3 树的过拟合和剪枝（pruning）

​    树的过拟合是一个非常严重的问题。在极端情况下，每个叶子节点可能只有一个元素，这样整棵树变成了一个look-up table. 这样的树几乎没有泛化能力！

​    为了解决决策树的过拟合问题，有两种方法 -**- pre-pruning 和 post-pruning**

1) pre-pruning: 当节点分类带来的收益不统计学上显著(statistically significant)时,停止分裂

- 方法1：如果到达该节点的样本数<总样本数*5%，则停止分裂。因为数据集过小会导致方差增大，从而导致泛化性变差。
- 方法2: 如果该节点分裂的信息增益<threshold，则停止分裂。

2）post-pruning：先生成整棵树，然后后剪枝

- 方法1 -- 减少误差的后剪枝方法(reduced-error post-pruning): 把数据集分为训练集和验证集，尝试剪掉每一个节点并测试在验证集上的结果。每次都贪心的剪掉那个带来最大验证集上准确率提升的节点。

> Split data into training set and validation set, do until further pruning is harmful: - Evaluate impact on validation set of pruning each possible node - Greedily remove the one that most improves validation set accuracy mostly.

- 方法2 -- 规则后剪枝(rule post-pruning)

![img](https://pic2.zhimg.com/v2-26aa11bb0ace0156849f030ef7727705_b.png)

1.4 ID3算法的缺点

- 信息增益准则对可**取值数目较多**的特征有所偏好(信息增益反映的给定一个条件以后不确定性减少的程度,必然是分得越细的数据集确定性更高)
- ID3算法没有进行决策树**剪枝**，容易发生过拟合
- ID3算法没有给出处理**连续**数据的方法，只能处理离散特征
- ID3算法不能处理带有**缺失值**的数据集,需对数据集中的缺失值进行预处理



### 0x02. C4.5 by Quinlan

针对如上所说的几个缺点，C4.5对它们进行了改进。

#### 1. 使用信息增益比代替信息增益

ID3中的信息增益其实是有bias的：对于那些特别多取值的特征(e.g. date,id)，信息增益自然就会很大。极端情况，如果直接用id作为一个特征，那么信息增益是很大的，因为按照id划分之后，节点不纯度变成了0！

所以，用信息增益比来代替信息增益。

![img](https://pic3.zhimg.com/v2-cb0c4f1a1ea7af20f0dfeba86e3203e6_b.png)

信息增益比对那些属性值特别多的属性(如date,id)做了惩罚。

但是，信息增益比对取值较少的特征有所偏好（分母越小，整体越大），因此 C4.5 并不是直接用信息增益比最大的特征进行划分，而是使用一个启发式方法：先从候选划分特征中**找到信息增益高于平均值的特征，再从中选择信息增益比最高的。**



#### 2. 使用后剪枝

在决策树构造完成后，自底向上对非叶节点进行评估，如果将其换成叶节点能提升泛化性能，则将该子树换成叶节点。后剪枝决策树欠拟合风险很小，泛化性能往往优于预剪枝决策树。但训练时间会大很多。



#### 3. 增加处理连续特征的方法 -- 连续特征离散化

- 需要处理的样本或样本子集按照连续变量的大小从小到大进行排序
- 假设该属性对应的不同的属性值一共有N个,那么总共有N−1个可能的候选分割阈值点,每个候选的分割阈值点的值为上述排序后的属性值中两两前后连续元素的中点,根据这个分割点把原来连续的属性分成两类

#### 4. 缺失值处理

- 对于具有缺失值的特征，用没有缺失的样本子集所占比重来折算信息增益率，选择划分特征
- 选定该划分特征，对于缺失该特征值的样本，将样本以不同的概率划分到不同子节点

##### 5. C4.5 的缺点

- C4.5只能用于分类
- C4.5 在构造树的过程中，对数值属性值需要按照其大小进行排序，从中选择一个分割点，所以只适合于能够驻留于内存的数据集，当训练集大得无法在内存容纳时，程序无法运行。而且，由于需要对数据集进行多次的顺序扫描和排序，算法低效。



### 3. CART算法(classification and regression)

   相比ID3和C4.5算法，CART算法使用**二叉树**简化决策树规模，提高生成决策树效率。CART决策树生成分为分类树和回归树两种场景，两者生成方式有一定的区别。

​    CART分类树在节点分裂时使用**GINI指数**来替代信息增益。基尼指数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。我们按照分裂前后基尼指数的差值，找到最好的分裂准则以及分裂值，将根节点一分为二。

​    例如，对于annual income和信贷风险的一个数据：

![img](https://pic1.zhimg.com/v2-a5e8aea9e5e29cac8dce753e18997400_b.png)

选择不同的位置做split，对于每个位置都衡量一下Gini impurity

### CART的优点：

- CART使用二叉树来代替C4.5的多叉树，提高了生成决策树效率
- C4.5只能用于分类，CART树可用于分类和回归(Classification And Regression Tree)
- CART 使用 Gini 系数作为变量的不纯度量，减少了大量的对数运算
- ID3 和 C4.5 只使用一次特征，CART 可多次重复使用特征

ID3、C4.5、CART对比：

![img](https://pic3.zhimg.com/v2-3982811f09d87ce4b84b2be27b849d5e_b.png)









------

参考资料：

- 清华大学2020春《机器学习概论》课件
- https://zhuanlan.zhihu.com/p/158633779








  