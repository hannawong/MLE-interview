## 0x01. 为什么要用多任务学习？

1. 方便。在推荐任务中，往往不仅要预测用户的engagement（例如CTR），还要预测用户satisfaction（例如评分、CVR、观看时长）。如果用多个模型预测多个目标，参数量会很大，而且在线上也不好维护。因此需要使用一个模型来预测多个目标，这点对工业界来说十分友好。
2. 多任务学习不仅方便，还可能效果更好。针对很多数据集比较稀疏的任务，比如短视频转发，大部分人看了一个短视频是不会进行转发这个操作的，这么稀疏的行为，模型是很难学好的（过拟合问题严重），那我们把预测用户是否转发这个稀疏的事情和用户是否点击观看这个经常发生事情放在一起学，通过参数共享，一定程度上会缓解模型的过拟合，提高了模型的泛化能力。这其实是regularization和transfer learning。也可以理解为，其他任务的预测loss对于"转发"事件预测来说是辅助loss。从另一个角度来看，对于数据很少的新任务，这样也解决了冷启动问题。

## 0x02. 多任务学习模型

### (1). Baseline -- Shared-Bottom Model

1.1 硬参数共享

![img](https://pic2.zhimg.com/v2-b6ea5e033525fe64a38efcb14335df29_b.png)

不同任务间共用底部的隐层。这种结构由于全部的参数共享可以减少过拟合的风险（原因如上所述），但是效果上受到任务差异（optimization conflicts caused by task differences）和数据分布差异带来的影响。

1.2 软参数共享

与硬参数共享相对的是软参数共享：每个任务都有特定的模型、参数，参数不共享；但对模型间的参数，使用距离正则化约束，保障参数空间的相似。

![img](https://pic2.zhimg.com/v2-d178f6657ce02686f15db3e7d6d0d18d_b.png)

两个任务的参数完全不共用，但是对不同任务的参数增加L2范数的限制（L2-Constrained）：

![img](https://pic4.zhimg.com/v2-91eef8071734d988f612d78d18e7f50b_b.png)

2个任务的参数完全不共用，但是在损失函数中加入正则项。α是两个任务的相似度，α越大，两个任务参数越趋近于一致。

和shared-bottom结构相比，这样的模型对增加了针对任务的特定参数(task-specific parameters)，在任务差异很大的情况下效果比较好。缺点就是模型增加了参数量（如果要训练k个目标，就增加k倍），所以需要更大的数据量来训练模型，而且模型更复杂并不利于在真实生产环境中实际部署使用。

（2）MMoE

论文 Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts 中提出了一个Multi-gate Mixture-of-Experts(MMoE)的多任务学习结构。

![img](https://pic4.zhimg.com/v2-1650f6ad8f96230f662483100abb4c17_b.png)

​                                                                   Shared-bottom, OMoE, MMoE

文章提出的模型MMoE目的就是相对于shared-bottom结构不明显增加模型参数 的要求下捕捉任务的不同 。其核心思想是将shared-bottom网络中的函数 f 替换成 MoE 层，如上图c所示，形式化表达为：

![f^k(x)=\sum_{i=1}^{n}{g^k(x)_if_i(x)} \\ y_k=h^k(f^k(x)),](https://www.zhihu.com/equation?tex=f%5Ek(x)%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bg%5Ek(x)_if_i(x)%7D%20%5C%5C%20y_k%3Dh%5Ek(f%5Ek(x))

其中门控网络 ![g^k(x)=softmax(W_{gk}x)](https://www.zhihu.com/equation?tex=g%5Ek(x)%3Dsoftmax(W_%7Bgk%7Dx)) ，输入就是input feature，输出是所有experts上的权重。其实这个门控很像attention，针对不同的任务分配给experts以不同的权重。

一方面，因为gating networks通常是轻量级的，而且expert network是所有任务共用，所以相对上文提到的软参数共享方法有参数量上的优势；

另一方面，相对于所有任务公用一个gate的方法One-gate MoE model(OMOE)，这里MMoE中每个任务使用不同的gating networks，从而学习到不同的组合experts的权重，因此模型考虑到了捕捉到任务的相关性和区别。因此在模型的效果上优于上文提到的硬参数共享的方法。实际上，如果任务相关度很低，则OMoE的效果相对于MMoE明显下降，说明MMoE中的multi-gate的结构对于任务差异带来的冲突有一定的缓解作用。

MMoE在Youtube推荐场景下的实践

论文：Recommending What Video to Watch Next: A Multitask Ranking System，这篇主要是在商业推荐上用了MMoE,以及提出了shallow tower解决position bias的方法。

文中的优化目标大体分为两类，一类是engagement目标，包括点击、观看时长、完播率等，表示用户的参与度；第二类是satisfaction目标，例如评分或差评，表示用户的满意度。这其中既有分类任务(e.g. clicked)也有回归任务(e.g. 观看时长、评分)。从文中实验来看，总共包括7个任务，这些任务或者是递进/依赖的关系，例如只有观看之后才会打分；或者是冲突的关系，点了之后发现不喜欢。MMoE比较适合这种多个任务之间联系不紧密、甚至冲突的场景。

完整的模型结构如下图所示。模型对每一个目标都做预估，分类问题就用cross entropy loss学习，回归问题可就是square loss。最后用融合公式来平衡用户交互和满意度指标（将目标预估结果做加权平均）取得最佳效果。这个权重需要人工手动来调整。

![img](https://pic3.zhimg.com/v2-5d3b129af147908228b87f7a21f2295e_b.png)

(3) SNR（Sub-Network-Routing）

出自论文 SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning。MMoE主要针对多个共享的expert 网络的输出进行attention组合(也就是门控)。SNR 在这种模块化的基础上，使用编码变量（coding variables）控制子网络之间的连接，实现多任务模型中不同程度的参数共享。SNR 的提出主要解决级间的参数共享问题，达到最佳组合的网络结构。简言之，SNR和MMoE的不同之处就是，MMoE拿多个子网络的输出做加权直接输入到了每个任务各自的tower中；而SNR对不同子网络的输出进行组合又输入到了下一层子网络，形成子网络的组合。

SNR设计了两种类型的连接方式：SNR-Trans 和 SNR-Aver来学习子网络的组合，最终得到特定多任务场景的最优网络结构。

![img](https://pic4.zhimg.com/v2-91aa0a4f16e8a2a4911cc192ae101bc3_b.jpeg)

图(b)(c)有两层子网络，分别是底层的DNN routing部分（3个sub-networks）和顶层的多任务tower部分（2个sub-networks）。

- SNR−Trans模型：将共享层划分为子网络，子网络之间的连接(虚线)为变换矩阵W乘以标量编码z:

![img](https://pic4.zhimg.com/v2-112403f818856542cd7d3b7a93db8af7_b.png)

其中，![ u_1,u_2,u_ 3 ](https://www.zhihu.com/equation?tex=%20u_1%2Cu_2%2Cu_%203%20) 代表low-level输出；![v_1,v_2 ](https://www.zhihu.com/equation?tex=v_1%2Cv_2%20) 代表high-level输入；![W_{i,j}](https://www.zhihu.com/equation?tex=W_%7Bi%2Cj%7D)代表 ![j](https://www.zhihu.com/equation?tex=j)low-level到 ![i](https://www.zhihu.com/equation?tex=i)i high-level的权重矩阵，![z_{i,j}](https://www.zhihu.com/equation?tex=z_%7Bi%2Cj%7D)z_{i,j} 代表 ![j](https://www.zhihu.com/equation?tex=j)j low-level到 ![i](https://www.zhihu.com/equation?tex=i)i high-level的连接性（二元变量，0 or 1）。

![W](https://www.zhihu.com/equation?tex=W)W和![z](https://www.zhihu.com/equation?tex=z)z是我们要学习的参数。假设![ z](https://www.zhihu.com/equation?tex=%20z) z服从 Bernoulli 分布，用0/1来控制网络的连接与断开。但是![z](https://www.zhihu.com/equation?tex=z)z是不可微的，把![z](https://www.zhihu.com/equation?tex=z)z做一个变换，变换成一个平滑的函数:

![img](https://pic3.zhimg.com/v2-a0ccf5f64770ee85be35283ef9d4cf9a_b.png)

- SNR−Aver模型：将共享层划分为子网络，子网络之间的连接(虚线)为加权平均。

![img](https://pic1.zhimg.com/v2-5f3a982c59884939e4cab4e8f22bd0fc_b.png)



(4) PLE（Progressive Layered Extraction）

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

------

下一篇将介绍多任务学习的目标loss设计和优化改进、一些辅助loss的设计、以及多目标排序的代码实践。

参考：

[被包养的程序猿丶：腾讯PCG RecSys2020最佳长论文——视频推荐场景下多任务PLE模型详解](https://zhuanlan.zhihu.com/p/272708728)

[Recommender：推荐系统中的多任务学习与多目标排序工程实践（上）](https://zhuanlan.zhihu.com/p/422925553)


  