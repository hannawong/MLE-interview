# 表征学习

## 1. 华为 | 一种CTR预估中连续特征的Embedding学习框架

#### 1.1 背景

当前大多数的研究主要集中在设计更复杂的网络架构来更好的捕获显式或隐式的特征交互。而另一个主要的部分，即Embedding模块同样十分重要，出于以下两个原因：

1）Embedding模块是FI模块的上游模块，直接影响FI模块的效果； 2）CTR模型中的大多数参数集中在Embedding模块(巨大的embedding table！)，对于模型效果有十分重要的影响。

但是，Embedding模块却很少有工作进行深入研究，特别是对于连续特征的embedding方面。现有的处理方式由于其硬离散化(hard discretization)的方式，通常suffer from low model capacity。而本文提出的AutoDis框架具有high model capacity, end-to-end training, 以及unique representation. 

- No Embedding：是指不对连续特征进行embedding操作，而直接使用原始的数值。如Google Play的Wide & Deep直接使用原始值作为输入;而在Youtube DNN 中，则是对原始值进行变换（如平方，开根号）后输入：

![img](https://pic1.zhimg.com/v2-f377e77b4c542178d32abd93d9861c28_b.png)

​        这类对连续特征不进行embedding的方法，由于模型容量有限，通常难以有效捕获连续特征中信息。

- Field Embedding：是指同一个field无论取何值，都共享同一个embedding，随后将特征值与其对应的embedding相乘作为模型输入：

![img](https://pic2.zhimg.com/v2-17a7c017d37624d7fed97dacc98bd60d_b.png)

​                     其中， $e_1,e_2,...e_n$ 是field embedding。

​        由于同一field的特征共享同一个embedding，并基于不同的取值对embedding进行缩放，这类方法的表达能力也是有限的。

- Discretization: 即将连续特征进行离散化，是工业界最常用的方法。这类方法通常是两阶段的，即首先将连续特征转换为对应的离散值，再通过look-up的方式转换为对应的embedding。

​       【为什么要将连续特征离散化呢？】将连续特征进行离散化给模型引入了**非线性**，能够提升模型表达能力，而对于离散化的方式，常用的有以下几种：

1） 等宽/等深分箱。对于等宽分箱，首先基于特征的最大值和最小值、以及要划分的桶的个数 ![H_j](https://www.zhihu.com/equation?tex=H_j)，来计算每个样本取值要放到哪个箱子里。对于等深分箱，则是基于数据中特征的**频次**进行分桶，每个桶内特征取值的个数是大致相同的。

2）LD (Logarithm Discretization)：对数离散化，其计算公式如下：

![img](https://pic2.zhimg.com/v2-217616388385baee2844c4a75dd9d9ed_b.png)

3）TD (Tree-based Discretization)：基于树模型的离散化，如使用GBDT+LR来将连续特征分到不同的节点。这就完成了离散化。

离散化方法的缺点： 1）**TPP (Two-Phase Problem)**：将特征分桶的过程一般使用启发式的规则（如EDD、EFD）或者其他模型（如GBDT），无法与CTR模型进行一起优化，即**无法做到端到端训练**； 2）SBD (Similar value But Dis-similar embedding)：对于边界值，两个相近的取值由于被分到了不同的桶中，导致其embedding可能相差很远； 3）DBS (Dis-similar value But Same embedding)：对于同一个桶中的边界值，两边的取值可能相差很远，但由于在同一桶中，其对应的embedding是完全相同的。

#### 1.2 AutoDis介绍

AutoDis的全称为Automatic end-to-end embedding learning framework for numerical features based on soft discretization. 用于连续特征的端到端离散化和embedding学习。

![img](https://pic4.zhimg.com/v2-d55666c6319dacfb6e541250b996a767_b.png)



##### 1.3.1 Meta-Embeddings

为了提升model capacity，一种朴素的处理连续特征的方式是给每一个特征取值赋予一个独立的embedding。显然，这种方法参数量巨大（因为你可以有无穷个连续特征取值！），无法在实践中进行使用。另一方面，Field Embedding对同一域内的特征赋予相同的embedding，尽管降低了参数数量，但model capacity也受到了一定的限制。

为了平衡参数数量和模型容量，AutoDis设计了Meta-embedding模块: **对于第 $j$ 个连续特征，对应 ![H_j](https://www.zhihu.com/equation?tex=H_j)个Meta-Embedding**（可以看作是分 ![H_j](https://www.zhihu.com/equation?tex=H_j)个桶，每一个桶对应一个embedding）。第$j$个特征的Meta-Embedding表示为：  

​                                                               ![ME_j \in \mathbb{R}^{H_j \times d}](https://www.zhihu.com/equation?tex=ME_j%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BH_j%20%5Ctimes%20d%7D)  ( ![H_j](https://www.zhihu.com/equation?tex=H_j)个桶，每个桶是d维的)

对于连续特征的一个具体取值，则是通过一定方式将这 ![H_j](https://www.zhihu.com/equation?tex=H_j)个embedding进行聚合。相较于Field Embedding这种每个field只对应一个embedding的方法，AutoDis中每一个field对应 ![H_j](https://www.zhihu.com/equation?tex=H_j)个embedding，提升了模型容量；同时，参数数量也可以通过 ![H_j](https://www.zhihu.com/equation?tex=H_j)进行很好的控制。

##### 1.3.2 Automatic Discretization

Automatic Discretization模块可以对连续特征进行自动的离散化，实现了离散化过程的端到端训练。具体来说，对于第 ![j](https://www.zhihu.com/equation?tex=j) 个连续特征的具体取值 ![x_j](https://www.zhihu.com/equation?tex=x_j) ，首先通过两层神经网络进行转换，得到 ![H_j](https://www.zhihu.com/equation?tex=H_j) 长度的向量。下图的例子假设有41个特征，每个特征分配 ![H_j = 10](https://www.zhihu.com/equation?tex=H_j%20%3D%2010) 个桶

![img](https://pic2.zhimg.com/v2-6fec12b115e734557d7495e4e935b8c9_b.jpeg)

最后得到的 ![\tilde{x_j}](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx_j%7D)需要经过某种softmax变成概率分布:

![img](https://pic3.zhimg.com/v2-8b413d778131652a5d9db0a092508762_b.png)

传统的离散化方式是将特征取值分到某一个具体的桶中，即对每个桶的概率进行argmax，但这是一种无法进行梯度回传的方式，是硬离散化。而上式可以看作是一种软离散化（soft discretization）。对于**温度系数**𝜏，当其接近于0时，得到的分桶概率分布接近于one-hot，当其接近于无穷时，得到的分桶概率分布近似于均匀分布。这种方式也称为softargmax。

至此，我们得到了 ![H_j](https://www.zhihu.com/equation?tex=H_j)个桶的embedding以及概率分布。

##### 1.3.3 Aggregation Function

其实就是对 ![H_j](https://www.zhihu.com/equation?tex=H_j)个桶的embedding进行加权求和。

## 

模型的训练过程同一般的CTR过程相似，采用二分类的logloss指导模型训练，损失如下：

![img](https://pic4.zhimg.com/v2-09bb6e99597b6cdec91eb176830e6887_b.jpg)



那么，AutoDis是否有效解决了SBD和DBS的问题呢？实验结果也印证了这一点：

![img](https://pic4.zhimg.com/v2-fa03d9510adc1132d0771fd436ad6ddf_b.png)

（右图：等深分箱，不同的取值都是分开的点，没有相似度的联系；左图：Autodis，相似的取值聚在一起，说明端到端的方法把握了数值的相似性）



## 2. 谷歌 | 不需要embedding table的类别特征embedding方法

*Learning to Embed Categorical Features without Embedding Tables for Recommendation*

#### 2.1 背景

对于类别型特征（用户ID/物品ID）标准的方式是用一个(巨大的) embedding table为每个类别特征分配一个embedding。然而这种方式有很大问题：

- 参数量巨大（Huge vocabulary size）：推荐系统通常包含几百万的用户ID/视频ID，如果每个特征都指定一个embedding会占据大量空间。
- 特征是动态的（Dynamic nature of input）：推荐系统中经常会出现**全新的**用户ID/视频ID，固定的embedding table不能解决**OOV**(out-of-vocabulary)问题.
- 特征分布高度倾斜（Highly-skewed data distribution）：推荐数据中**低频**特征的训练实例数量较少，因此该特征的embedding在训练阶段就很少更新，对训练的质量有显著影响。

**已有的类别特征embedding方法**：

- **One-hot Full Embedding**：这种方式就是最常见的方法，做one-hot encoding，然后通过一个可学习的线性变换矩阵（说白了就是embedding table，可以看作一层神经网络，但没有bias项）得到对应的embedding表示： ![\textbf{e} = W^T\textbf{b}](https://www.zhihu.com/equation?tex=%5Ctextbf%7Be%7D%20%3D%20W%5ET%5Ctextbf%7Bb%7D)。缺点：embedding table随特征数量线性增长（即内存问题）；无法处理新出现的特征（OOV）。
- **One-hot Hash Embedding**：为了解决One-hot Full Embedding中的内存消耗巨大的问题，可以使用**哈希**函数对类别特征进行映射分桶，将原始的 ![n](https://www.zhihu.com/equation?tex=n)维的 one-hot 特征编码映射为m维的 one-hot 特征编码(即m个桶)。这样，embedding table只用存储m项，大大降低了参数量。缺点：只要是哈希，就会有冲突！哈希冲突导致多个ID共用一个embedding, 这会伤害模型性能。

为了解决哈希冲突的问题，可以做如下改进：

取k个不同的哈希函数 ![\{H^{(1)},H^{(2)},...H^{(k)}\}](https://www.zhihu.com/equation?tex=%5C%7BH%5E%7B(1)%7D%2CH%5E%7B(2)%7D%2C...H%5E%7B(k)%7D%5C%7D) , 按照上述方法生成**k**个one-hot编码： ![\{\textbf{b}^{(1)},{\textbf{b}^{(2)}},...{\textbf{b}^{(k)}}\}](https://www.zhihu.com/equation?tex=%5C%7B%5Ctextbf%7Bb%7D%5E%7B(1)%7D%2C%7B%5Ctextbf%7Bb%7D%5E%7B(2)%7D%7D%2C...%7B%5Ctextbf%7Bb%7D%5E%7B(k)%7D%7D%5C%7D)(分到了k个桶), 每个one-hot编码都去查一下embedding table，并且把最后的结果concat到一起/或者做avg-pooling。

#### 2.2 Deep Hash Embeddings

DHE将整个特征嵌入分为编码阶段(encoding)和解码阶段(decoding)。下图是One-hot Embedding与DHE的整体区别：

![img](https://pic1.zhimg.com/v2-260199fcfb09a85f7487e58e10b03064_b.png)

可以看到：

- DHE编码阶段通过多个(k=1024个)哈希函数将特征表示为**稠密的Identifier vector**, 解码阶段通过多层神经网络得到该特征的唯一表示。

##### 2.2.1 Encoding阶段

**【一个好的encoding应该有哪些特性】**

- 唯一性（Uniqueness）：每个不同特征值的编码应该是唯一的。

- 同等相似性（ Equal Similarity）：只有唯一表示是不够的。例如二进制编码中：9表示为1001，8表示为1000：，7表示为0111。我们发现8的表示和9的表示更相似，这会引入bias，让编码器以为id = 8 与 id = 9比起id = 7 与 id = 9更相似，然而id类特征是没有顺序的，因此它们应该是**同等相似的**。
- 高维（High dimensionality）：我们希望这些编码便于后续解码函数区分不同的特征值。由于高维空间通常被认为是更可分的 (回忆一下SVM中的kernel方法...)，我们认为编码维度也应该相对较高。
- 高香农熵（High Shannon Entropy）：香农熵(以bit为单位)测量一个维度中所携带的信息。从信息论的角度来看，高香农熵的要求是为了防止冗余维数。例如，一个编码方案可能满足上述三个属性，但在某些维度上，所有特征值的编码值是相同的。所以我们希望通过最大化每个维度的熵来有效地利用所有维度。例如，one-hot编码在每个维度上的熵都很低，因为对于大多数特征值来说，每个维度上的编码都是0。因此，one-hot编码需要非常高的维度(即)，而且效率非常低。高香农熵就是要让encoding更加高效。

**DHE编码阶段(encoding)的设计：**

提出的DHE运用 ![k](https://www.zhihu.com/equation?tex=k) 个哈希函数把每个类别特征映射为一个 ![k](https://www.zhihu.com/equation?tex=k) 维的稠密向量。

具体的，每个哈希函数 ![H^{(i)}](https://www.zhihu.com/equation?tex=H%5E%7B(i)%7D) 都将一个正整数 ![\mathbb{N}](https://www.zhihu.com/equation?tex=%5Cmathbb%7BN%7D) 映射到${1,2,...m}$,本实验中取 ![m = 1e6](https://www.zhihu.com/equation?tex=m%20%3D%201e6) 。因此，k个哈希函数就把一个正整数 ![\mathbb{N}](https://www.zhihu.com/equation?tex=%5Cmathbb%7BN%7D)映射成了k维的向量 ![E'(s) = [H^{(1)}(s),H^{(2)}(s),...,H^{(k)}(s)]](https://www.zhihu.com/equation?tex=E%27(s)%20%3D%20%5BH%5E%7B(1)%7D(s)%2CH%5E%7B(2)%7D(s)%2C...%2CH%5E%7B(k)%7D(s)%5D), 向量中的每个元素都取自{1,2,...m}。实验中取k=1024.

然而，直接用上面得到的编码表示是不合适的，因此作者进行了两个变换操作来保证数值稳定性：

- 均匀分布（Uniform Distribution）：把 ![E'(s)](https://www.zhihu.com/equation?tex=E%27(s))E'(s) 中的每个值映射到[-1,1]之间
- 高斯分布（Gaussian Distribution）：把经过均匀分布后的向量转化为高斯分布 ![N(0,1)](https://www.zhihu.com/equation?tex=N(0%2C1))N(0,1) 。

(作者说，这里是受到GAN网络的启发，用服从高斯分布的随机变量做GAN网络的输入。)

作者在文章中验证了这样设计的encoding满足上述的四个条件。

##### 2.2.2 Decoding 阶段

Decoding阶段需要把Encoding阶段得到的 ![k](https://www.zhihu.com/equation?tex=k) 维向量映射为 ![d](https://www.zhihu.com/equation?tex=d) 维。如上面的图所示，作者用**多层神经网络**来实现。但是，由于参数量明显比embedding table降低很多，这种多层神经网络会导致**欠拟合**。因此，作者尝试了Mish激活函数 ![f(x) = x·tanh(ln(1+e^x))](https://www.zhihu.com/equation?tex=f(x)%20%3D%20x%C2%B7tanh(ln(1%2Be%5Ex))) 来代替ReLU激活函数，**引入更多的非线性**，从而提升表征能力。

作者还考虑了batch normalization (BN)等训练技巧。但是不能使用dropout，因为我们的问题是欠拟合而不是过拟合。



#### 2.3 加入辅助信息以增强【泛化性】(解决OOV问题)

- 记忆性(memorization): 例如one-hot编码的每个id embedding都是独立的，因此只有记忆性没有泛化性
- 泛化性(generalization): 本文提出的DHE方法，embedding network中的参数变化会影响所有特征的embedding结果。

对于物品ID/用户ID的特征embedding，可以考虑拼接上它们属性（年龄、品牌等）的表示，然后输入到DHE解码阶段来生成最终的特征嵌入。这样能够增强泛化能力。



结果：

- DHE取得了和one-hot相近的性能，但参数了极大的减小了。



  