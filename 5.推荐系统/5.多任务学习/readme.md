(3) SNR（Sub-Network-Routing）

出自论文 SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning。MMoE主要针对多个共享的expert 网络的输出进行attention组合(也就是门控)。SNR 在这种模块化的基础上，使用编码变量（coding variables）控制子网络之间的连接，实现多任务模型中不同程度的参数共享。SNR 的提出主要解决级间的参数共享问题，达到最佳组合的网络结构。简言之，SNR和MMoE的不同之处就是，MMoE拿多个子网络的输出做加权直接输入到了每个任务各自的tower中；而SNR对不同子网络的输出进行组合又输入到了下一层子网络，形成子网络的组合。

SNR设计了两种类型的连接方式：SNR-Trans 和 SNR-Aver来学习**子网络的组合**，最终得到特定多任务场景的最优网络结构。

![img](https://pic4.zhimg.com/v2-91aa0a4f16e8a2a4911cc192ae101bc3_b.jpeg)

图(b)(c)有两层子网络，分别是底层的DNN routing部分（3个sub-networks）和顶层的多任务tower部分（2个sub-networks）。

- SNR−Trans模型：将共享层划分为子网络，子网络之间的连接(虚线)为变换矩阵W乘以标量编码z:

![img](https://pic4.zhimg.com/v2-112403f818856542cd7d3b7a93db8af7_b.png)

其中，![ u_1,u_2,u_ 3 ](https://www.zhihu.com/equation?tex=%20u_1%2Cu_2%2Cu_%203%20) 代表low-level输出；![v_1,v_2 ](https://www.zhihu.com/equation?tex=v_1%2Cv_2%20) 代表high-level输入；![W_{i,j}](https://www.zhihu.com/equation?tex=W_%7Bi%2Cj%7D)代表 ![j](https://www.zhihu.com/equation?tex=j)low-level到 ![i](https://www.zhihu.com/equation?tex=i)i high-level的权重矩阵，![z_{i,j}](https://www.zhihu.com/equation?tex=z_%7Bi%2Cj%7D)z_{i,j} 代表 ![j](https://www.zhihu.com/equation?tex=j)j low-level到 ![i](https://www.zhihu.com/equation?tex=i)i high-level的连接性（二元变量，0 or 1）。

![W](https://www.zhihu.com/equation?tex=W)W和![z](https://www.zhihu.com/equation?tex=z)z是我们要学习的参数。假设![ z](https://www.zhihu.com/equation?tex=%20z) z服从 Bernoulli 分布，用0/1来控制网络的连接与断开。但是![z](https://www.zhihu.com/equation?tex=z)z是不可微的，把![z](https://www.zhihu.com/equation?tex=z)z做一个变换，变换成一个平滑的函数:

![img](https://pic3.zhimg.com/v2-a0ccf5f64770ee85be35283ef9d4cf9a_b.png)

- SNR−Aver模型：将共享层划分为子网络，子网络之间的连接(虚线)为加权平均。

![img](https://pic1.zhimg.com/v2-5f3a982c59884939e4cab4e8f22bd0fc_b.png)



**(4) PLE（Progressive Layered Extraction）**

RecSys2020最佳长论文 Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations.

文章首先提出多任务学习中不可避免的两个缺点：

- Negative Transfer. 针对相关性较差的多任务来说，使用hard parameter sharing这种方式通常会出现negative transfer的现象，原因就是因为任务之间相关性较低或者说存在冲突的话，会导致模型无法有效进行参数的学习，学的不伦不类。
- 跷跷板现象。对于一些任务相关性比较复杂的场景，通常会出现跷跷板现象，即提升一部分任务的同时，会牺牲其他任务的效果。

为了解决“跷跷板”现象，文章针对多任务之间的共享机制和单任务的特定网络结构进行了重新的设计，提出了PLE模型。

1) CGC (custom gate control)

首先，只考虑一层的抽取网络，就是CGC。

![img](https://pic1.zhimg.com/v2-38202e18b159bb2e3650fe356daf0e54_b.png)

从图中的网络结构可以看出，CGC的底层网络主要包括shared experts和task-specific experts构成，每一个expert module都由多个expert子网络组成。每个子任务tower的输入是对task-specific和shared两部分的expert vector进行加权求和。

CGC网络的好处是既包含了task-specific网络独有的个性化信息，也包含了shared 网络具有的更加泛化的信息，文章指出虽然MMoE模型在理论上可以得到同样的解，但是在实际训练过程中很难收敛到这种情况。

2) PLE

PLE就是上述CGC网络的多层叠加，以获得更加丰富的表征能力。具体网络结构如下图所示：

![img](https://pic2.zhimg.com/v2-77cf009231a3e2a83323f8dd815f70cd_b.png)

注意，在底层的Extraction网络中，除了各个task-specifict的门控网络外，还有一个share部分的门控网络，这部分门控网络的输入包含了所有input，而各个task-specific的门控网络的输入是task-specific和share两部分。



### 梯度优化

###### Project Conflicting Gradients (PCGrad) [斯坦福，NIPS2020]

出自论文 Gradient Surgery for Multi-Task Learning，这个名字十分形象："Gradient Surgery".

在多任务训练期间，如果能知道具体的梯度就可以利用梯度来动态更新 ![[公式]](https://www.zhihu.com/equation?tex=w) 。 如果两个任务的梯度存在**冲突**（即余弦相似度为负），将任务A 的梯度投影到任务B 梯度的法线上。**即是消除任务梯度的冲突部分**，减少任务间冲突。同时，对模长进行了归一化操作，防止梯度被某个特别大的主导了。

![img](https://pic3.zhimg.com/v2-ffaccdc4ffe5fec9e85f000475153ca2_b.jpg)多任务优=

1. 首先通过计算 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 与 ![[公式]](https://www.zhihu.com/equation?tex=g_j) 之间的余弦相似度来判断 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 是否与 ![[公式]](https://www.zhihu.com/equation?tex=g_j) 冲突；其中负值表示梯度冲突。
2. 如果余弦相似度是负数，我们用它在 ![[公式]](https://www.zhihu.com/equation?tex=+g+) 的法线平面上的投影替换。如果梯度不冲突，即余弦相似度为非负，原始梯度 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 保持不变。

![[公式]](https://www.zhihu.com/equation?tex=g_j%3Ag_i+%3D+g_i+-+%5Cfrac%7Bg_i+g_j%7D%7B%7C%7Cg_j%7C%7C%5E2%7D%5C%5C)

其实，这篇文章是很有问题的。梯度冲突不一定是个坏事，而可以带来正则化的好处，如果只是把冲突梯度完全抹去恐有不妥。而且，文章并没有对梯度裁剪进行消融实验。万一是梯度裁剪、而不是Project conflict 导致的性能提升呢？不过，这个方法在我的测试上也是表现比原来的好的。

参考：

[被包养的程序猿丶：腾讯PCG RecSys2020最佳长论文——视频推荐场景下多任务PLE模型详解](https://zhuanlan.zhihu.com/p/272708728)

[Recommender：推荐系统中的多任务学习与多目标排序工程实践（上）](https://zhuanlan.zhihu.com/p/422925553)

