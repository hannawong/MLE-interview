

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



参考：

[被包养的程序猿丶：腾讯PCG RecSys2020最佳长论文——视频推荐场景下多任务PLE模型详解](https://zhuanlan.zhihu.com/p/272708728)

[Recommender：推荐系统中的多任务学习与多目标排序工程实践（上）](https://zhuanlan.zhihu.com/p/422925553)

