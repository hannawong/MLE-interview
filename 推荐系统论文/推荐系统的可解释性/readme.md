# GLIDER - 发现重要交叉特征

Global Interaction Detection and Encoding for Recommendation(GLIDER) 是一种发现神经网络学习到的任意阶的**交叉特征**的方法。主要包含以下两篇论文：

1. Detecting Statistical Interactions from Neural Network Weights (ICLR'18)
2. Feature Interaction Interpretability: A Case for Explaining Ad-Recommendation Systems via Neural Interaction Detection (ICLR'19)

第一篇文章提出一种**交叉特征检测的方法 (NID)**，主要是发现 MLP 学习到的比较重要的交叉特征。第二篇文章把 NID 方法用到推荐模型上，去发现推荐模型学习到的交叉特征。之后，再把发现的重要后验交叉特征加到原始的模型上，然后重新训练模型提升模型效果。

### 1. Neural Interaction Detection (NID) -- 发现MLP的重要交叉特征

如果把整个网络看成一个有向无环图，输入层每个特征和中间隐层的神经元看成图的节点，连接节点之间的权重看成边。对任何一个交叉特征集合 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BI%7D) ，都一定存在一个节点 ![[公式]](https://www.zhihu.com/equation?tex=V_%7B%5Cmathcal%7BI%7D%7D) 是他们共同的子孙。基于这个想法，第一篇文章把MLP第一个隐层的所有节点看作是我们需要找的“共同子孙” ![[公式]](https://www.zhihu.com/equation?tex=V_%7B%5Cmathcal%7BI%7D%7D) 。那么在第一层的第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个神经元上，交叉特征 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BI%7D) 的强度(interaction strength)记为![[公式]](https://www.zhihu.com/equation?tex=%5Comega_i%28%5Cmathcal%7BI%7D%29)。整个模型交叉特征 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BI%7D) 的组合强度则是把 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega_i%28%5Cmathcal%7BI%7D%29) 累加起来，记为 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega%28%5Cmathcal%7BI%7D%29)

![[公式]](https://www.zhihu.com/equation?tex=%5CLarge%5Cbegin%7Bsplit%7D%5Comega%28%5Cmathcal%7BI%7D%29+%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bp_1%7D%5Comega_i%28%5Cmathcal%7BI%7D%29%5C%5C+%5Comega_i%28%5Cmathcal%7BI%7D%29+%26%3D+z_i%5E%7B%281%29%7D%5Ccdot%5Cmu%28%7C%5Cmathbf%7Bw%7D_%7Bi%2C%5Cmathcal%7BI%7D%7D%5E%7B%281%29%7D%7C%29%5Cend%7Bsplit%7D+%5C%5C)

从上面式子可以看出，交叉特征的强度是神经元 ![[公式]](https://www.zhihu.com/equation?tex=i) 前面部分( ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%28%7C%5Cmathbf%7Bw%7D_%7Bi%2C%5Cmathcal%7BI%7D%7D%5E%7B%281%29%7D%7C%29) )和后面部分( ![[公式]](https://www.zhihu.com/equation?tex=z_i%5E%7B%281%29%7D) )的乘积，![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%28%7C%5Cmathbf%7Bw%7D_%7Bi%2C%5Cmathcal%7BI%7D%7D%5E%7B%281%29%7D%7C%29)是交叉特征 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BI%7D) 与神经元 ![[公式]](https://www.zhihu.com/equation?tex=i) 连接的权重![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D_%7B%5Cmathcal%7BI%7D%7D) 的某种均值函数（实际取的是min）,后面部分![[公式]](https://www.zhihu.com/equation?tex=z_i%5E%7B%281%29%7D)是神经元 ![[公式]](https://www.zhihu.com/equation?tex=i) 对最终预测 ![[公式]](https://www.zhihu.com/equation?tex=y) 的影响，或者说是神经元 ![[公式]](https://www.zhihu.com/equation?tex=i) 的重要度（参考图1 示意图）。

![img](https://pic4.zhimg.com/80/v2-b6cc2f239153fa042a879dbcd105b20f_1440w.png)图1 

第一篇文章用了**权重矩阵连乘**来作为神经元 ![[公式]](https://www.zhihu.com/equation?tex=i) 的重要度![[公式]](https://www.zhihu.com/equation?tex=z_i%5E%7B%281%29%7D)，它是模型输出 ![[公式]](https://www.zhihu.com/equation?tex=y) 对神经元 ![[公式]](https://www.zhihu.com/equation?tex=i) 输出梯度绝对值的上界。这个不等式的证明参考原文的附录C. 为啥要证明它是梯度绝对值的上界，是因为梯度绝对值是一种常见的重要度度量方案，这里NID**用权重矩阵连乘来近似**。

![[公式]](https://www.zhihu.com/equation?tex=%5CLarge+z%5E%7B%281%29%7D+%3D+%7B%7C%5Cmathbf%7Bw%7D%5Ey%7C%7D%5ET%5Ccdot%5Cprod_%7Bl%3DL%7D%5E%7B2%7D%7C%5Cmathbf%7Bw%7D%5E%7B%28l%29%7D%7C+%5C%5C+%5CLarge+z_i%5E%7B%281%29%7D%5Cgeq+%7C%5Cfrac%7B%5Cpartial+y%7D%7B%5Cpartial+h_i%5E%7B%281%29%7D%7D+%7C%5C%5C)

#### Greedy Ranking

 greedy ranking，对每个神经元不进行全量的![[公式]](https://www.zhihu.com/equation?tex=2%5Ep-2)个特征组合遍历，而是每次取 top n 的特征组合，那么特征组合的数量由![[公式]](https://www.zhihu.com/equation?tex=O%282%5Ep%29%5Crightarrow+O%28p%29)，这个过程可以参考图3的动图。

![img](https://pic4.zhimg.com/80/v2-92b2199eb57c16b196bf54ffd84785d7_1440w.jpg)

注意中间是两层循环，每次都去只更新前top p个重要交叉特征的强度：

![img](https://pic3.zhimg.com/v2-9fda188a0455b715215f1f0b095b09ee_b.webp)





## 2. GLIDER

NID只能发现 **MLP** 的交叉特征，那么我们的推荐模型如果不是MLP怎么样呢？第二篇文章结合 **LIME** 方法学习一个**局部**代理模型(MLP)，然后再使用 NID 去发现这个代理模型的交叉特征。

这里简单说下 LIME 扰动数据的思路：给一个样本![[公式]](https://www.zhihu.com/equation?tex=x)，我们可以随机改变它的某一维特征值，对于实值类型则置为默认值（例如0），这样就能得到一个新样本![[公式]](https://www.zhihu.com/equation?tex=x%27)，重复 n 次就能根据一个样本生成 n 个样本。这些样本都是分布在原始样本的附近，那么可以用分类模型![[公式]](https://www.zhihu.com/equation?tex=f_%7Brec%7D)（MLP）对这些样本进行预测，这样就能构成一个新的数据集![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BD%7D_%7Bp%7D%3D%5C%7B%3Cx%5E%7B%27%7D%2Cy%5E%7B%27%7D%3Df_%7Brec%7D%28x%5E%7B%27%7D%29%3E%7Cx%5E%7B%27%7D%5Cin+%5Ctext%7Bperturbate%7D%28x%29%5C%7D)

然后在这个新的数据集上的训练一个MLP，并用NID去检测这个 MLP 的交叉特征。到目前为止得到的交叉特征可以看作是 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Brec%7D) 的**局部代理模型**（毕竟 MLP 是在 x 附近的点上训练得到的）。第二篇文章中提出针对推荐模型的全局特征检测方法：随机选取![[公式]](https://www.zhihu.com/equation?tex=N)个点，重复上面用代理模型做交叉特征检测的操作，然后把这 N 次结果得到的特征组合累计。这就是“Global”的由来。

![img](https://pic4.zhimg.com/80/v2-4059606c689483e8d52df0d3ead29eaf_1440w.jpg)



最后，把发现的重要交叉特征再次加入模型中。





## 2. Ante-hoc可解释性

使用注意力机制，例如FiBiNET, InterHAt, AutoInt



