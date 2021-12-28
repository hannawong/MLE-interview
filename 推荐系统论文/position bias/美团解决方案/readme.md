# 美团： 解决position bias

之前去除position bias的方法：主要就是把位置作为神经网络中的特征/模块，放于网络的Wide部分，在线下训练时使用真实位置；但是，由于在预估过程中并不知道真实位置信息，所以在线上预估时使用固定位置。这种方法由于其简单性和有效性，在工业界被广泛应用。例如，为了在线上预估时无需使用位置信息，PAL将样本的CTR建模为ProbSeen乘以pCTR，其中ProbSeen仅使用位置特征建模，而pCTR使用其他信息建模，在线上只使用pCTR作为CTR预估值。

但是，这种方法有两个缺点:

1. 训练和预估之间位置信息的不同处理方法，导致**线下线上间的不一致**问题。
2. "用户是否查看item只和item的位置有关"(PAL中的假设)--这个假设对问题过于简化了。事实是，**不同的用户通常具有不同的浏览习惯**：有些用户可能倾向于浏览更多item，而有些用户通常能快速做出决定；而且同一个用户在不同的**上下文搜索意图**中也会有不同的位置偏好，例如商场等地点词的搜索往往意图不明确导致高低位置的CTR差异并不大。故而，**位置偏差与用户、上下文都有关**，甚至可能与广告本身也有关，建模它们间的关系能更好地解决位置偏差问题。

美团提出的深度位置交叉网络能够较好的解决这个问题：

https://arxiv.org/pdf/2106.05482.pdfarxiv.org



## 1. 深度位置交叉网络（Deep Position-wise Interaction Network）

DPIN模型由三个模块组成：

- 处理 J 个**候选广告**的基础模块（Base Module）
- 处理 K 个**候选位置**的深度位置交叉模块（Deep Position-wise Interaction Module）
- 组合 J 个**广告和** K 个**位置**的位置组合模块（Position-wise Combination Module）

不同模块需预估的样本数量不一样，复杂模块预估的样本数量少，简单模块预估的样本数量多。通过这三个模块的组合，DPIN模型可以预估**每个广告在每个位置上的CTR**：![CTR_k^j](https://www.zhihu.com/equation?tex=CTR_k%5Ej)是第 j 个广告在第 k 个位置的CTR预估值，广告的最终序可以通过最大化![\Sigma{CTR^j_kbid^j}](https://www.zhihu.com/equation?tex=%5CSigma%7BCTR%5Ej_kbid%5Ej%7D)来确定，其中![bid^j](https://www.zhihu.com/equation?tex=bid%5Ej)为广告的出价。

### 1.1 基础模块(Base Module)

得到所有 ![J](https://www.zhihu.com/equation?tex=J) 个广告的embedding，使用简单的Embedding+MLP结构：

![img](https://pic3.zhimg.com/v2-965e9b28320a7853fb47cbe5867faf86_b.jpeg)

![img](https://pic2.zhimg.com/v2-ba5ecac3400daaee02bc394529d7cad9_b.jpeg)

![u_i,...,u_m,  c_1,...,c_m , i^j_1,...,i^j_o](https://www.zhihu.com/equation?tex=u_i%2C...%2Cu_m%2C%20%20c_1%2C...%2Cc_m%20%2C%20i%5Ej_1%2C...%2Ci%5Ej_o) 分别是当前用户特征集合、当前上下文特征集合以及第 j 个广告的特征集合。最终得到所有 ![J ](https://www.zhihu.com/equation?tex=J%20)  个广告在当前用户、当前上下文中的embedding表示。

### 1.2 深度位置交叉模块（Deep Position-wise Interaction Module）

在上面这一步，我们已经完成了**所有广告与user特征和context特征的交叉**；我们还需要完成所有的 ![J](https://www.zhihu.com/equation?tex=J) 个广告与所有 ![K](https://www.zhihu.com/equation?tex=K) 个位置的交叉。如果直接把位置特征放在Base Module中，就要完成 ![J*K](https://www.zhihu.com/equation?tex=J*K) 次计算。而在大多数业务场景中，Base Module通常已经被高度优化，包含了大量特征甚至用户序列等信息，所以这样做复杂度太高了。因此，我们需要一个深度位置交叉模块，来专门建模不同位置信息。

为了得到**不同位置在当前context、当前user下的embedding**，使用了context 特征和**用户在第 k 个位置的历史行为序列**：![B_k = b^k_1,b^k_2,...,b^k_L](https://www.zhihu.com/equation?tex=B_k%20%3D%20b%5Ek_1%2Cb%5Ek_2%2C...%2Cb%5Ek_L)，其中![b^k_l = [v^k_l,c^k_l]](https://www.zhihu.com/equation?tex=b%5Ek_l%20%3D%20%5Bv%5Ek_l%2Cc%5Ek_l%5D)为用户在第 ![k](https://www.zhihu.com/equation?tex=k) 个位置上的历史第 ![l](https://www.zhihu.com/equation?tex=l) 个行为记录，![v_l](https://www.zhihu.com/equation?tex=v_l)为点击的item特征，![c^k_l ](https://www.zhihu.com/equation?tex=c%5Ek_l%20) 为发生该行为时的context特征（包括搜索关键词、请求地理位置、一周中的第几天、一天中的第几个小时等）。这些行为序列和当前上下文 context 去计算注意力权重，对于与上下文越相关的行为可以给予越多的权重。

![img](https://pic3.zhimg.com/v2-be8b46bde4050487e0290bf211cc867a_b.png)

为了获得用户在其他位置上的行为序列信息，采用Transformer去学习不同位置兴趣的**交互**，最后得到K个输出，其中第 k 个位置被表示为![r^{pos}_k](https://www.zhihu.com/equation?tex=r%5E%7Bpos%7D_k)。

![img](https://pic1.zhimg.com/v2-d93b98deb646994b5306d17366c5ffb0_b.jpeg)

### 1.3 位置组合模块（Position-wise Combination Module）

位置组合模块的目的是去组合 J 个广告和 K 个位置来预估每个广告在每个位置上的CTR. 把Base Module输出的J个广告embedding（包含了user，context特征交叉）和深度位置交叉模块输出的K个位置embedding(包含user，context特征交叉)输出到一个MLP中，得到J * K大小的预估矩阵。

整个模型可以使用真实位置通过批量梯度下降法进行训练，采用交叉熵作为损失函数。

## 2.实验

- DIN： 没有做position bias的消除
- DIN+PosInWide： 在网络的Wide部分加入位置特征进行训练，在测试时位置特征取默认值。
- DIN+PAL： 采用PAL框架去建模位置信息。
- DIN+ActualPosInWide： 在网络的Wide部分加入位置特征进行训练，在测试时采用真实位置特征。
- DIN+Combination： 这个方法在DIN的基础上添加了位置组合模块，测试时采用真实位置特征。
- DPIN-Transformer： 在DPIN模型上去除了Transformer结构，来验证Transformer的作用。
- DPIN： DPIN模型。
- DPIN+ItemAction： 在DPIN的Base Module MLP层前添加深度位置交叉模块，并在位置兴趣聚合和位置非线性交叉中引入候选广告的信息，这个实验是DPIN方法模型性能的**理论上界**(因为在Base Module和深度位置交叉模块都做了候选item和position的交互)，然而服务性能是不可接受的。

![img](https://pic4.zhimg.com/v2-3db55784f6c46dfdfd86bfb250f45d4b_b.png)

A/B测试表明，DPIN在CTR上提高了2.25％，在RPM（每千次展示收入）上提高了2.15％。

------

参考：

[Yuki：SIGIR 2021 - 广告系统位置偏差的CTR模型优化方案［美团］- Deep Position-wise Interaction Network](https://zhuanlan.zhihu.com/p/380247607)




  