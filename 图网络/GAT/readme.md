

# GAT (Graph Attention Networks)

### 1. GAT的具体做法

对于每个节点，注意力其在邻居顶点上的注意力。对于顶点 ![[公式]](https://www.zhihu.com/equation?tex=i) ，逐个计算它的邻居们（ ![[公式]](https://www.zhihu.com/equation?tex=j+%5Cin+%5Cmathcal%7BN%7D_i) ）和它自己之间的**相似系数**：

![[公式]](https://www.zhihu.com/equation?tex=e_%7Bij%7D+%3Da%5Cleft%28+%5Cleft%5B+Wh_i+%5Cbig%7C+%5Cbig%7C++Wh_j+%5Cright%5D++%5Cright%29%2Cj+%5Cin+%5Cmathcal%7BN%7D_i+%5Cqquad+%281%29)

首先一个共享参数 ![[公式]](https://www.zhihu.com/equation?tex=W) 的线性映射对于顶点的特征进行了增维，当然这是一种常见的特征增强（feature augment）方法；![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5B+%5Ccdot+%5Cbig%7C+%5Cbig%7C+%5Ccdot%5Cright%5D+) 对于顶点 ![[公式]](https://www.zhihu.com/equation?tex=i%2Cj) 的变换后的特征进行了拼接（concatenate）；最后 ![[公式]](https://www.zhihu.com/equation?tex=a%28%5Ccdot%29) 把拼接后的高维特征映射到一个实数上，作者是通过单层的MLP来实现的。

然后，再对此相关系数用softmax做归一化：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7Bij%7D%3D%5Cfrac%7Bexp%5Cleft%28+LeakyReLU%28e_%7Bij%7D%29+%5Cright%29%7D%7B%5Csum_%7Bk%5Cin+%5Cmathcal%7BN%7D_i%7D%7Bexp%5Cleft%28+LeakyReLU%28e_%7Bik%7D%29+%5Cright%29%7D%7D+%5Cqquad+%282%29)

最后，根据计算好的注意力系数，把特征加权求和一下。这也是一种aggregation，只是和GCN不同，这个aggregation是带注意力权重的。

![[公式]](https://www.zhihu.com/equation?tex=h_i%5E%7B%27%7D%3D%5Csigma%5Cleft%28+%5Csum_%7Bj%5Cin+%5Cmathcal%7BN%7D_i%7D%7B%5Calpha_%7Bij%7DW%7Dh_j+%5Cright%29+%5Cqquad+%283%29)

$h_i'$就是输出的节点$i$的embedding，融合了其邻居和自身带注意力的权重（这里的注意力是self-attention）。

为了增强特征提取能力，用multi-head attention来进化增强一下：

![[公式]](https://www.zhihu.com/equation?tex=h_i%5E%7B%27%7D%28K%29%3D+%5Coverset%7BK%7D%7B%5Cunderset%7Bk%3D1%7D%7B%5Cbig%7C+%5Cbig%7C%7D%7D++%5Csigma%5Cleft%28+%5Csum_%7Bj%5Cin+%5Cmathcal%7BN%7D_i%7D%7B%5Calpha_%7Bij%7D%5Ek+W%5Ek%7Dh_j+%5Cright%29+%5Cqquad+%284%29)

下图中，有三个multi-head，$h_1'$是节点1的aggregated表示。

![img](https://pic3.zhimg.com/80/v2-226a97f07f1352e741eaeefdec6044ce_1440w.jpg)



### 2. 与GCN的联系

- GCN与GAT都是将邻居顶点的特征聚合到中心顶点上（一种aggregate运算）。不同的是GCN利用了拉普拉斯矩阵，GAT利用attention系数。一定程度上而言，GAT会更强，因为 顶点特征之间的相关性被更好地融入到模型中。
- GAT适用于有向图。这是因为GAT的运算方式是逐顶点的运算（node-wise），每一次运算都需要**循环遍历图上的所有顶点**来完成。逐顶点运算意味着，摆脱了拉普利矩阵的束缚，使得有向图问题迎刃而解。

- GAT适用于inductive任务。GAT中重要的学习参数是 ![[公式]](https://www.zhihu.com/equation?tex=W) 与 ![[公式]](https://www.zhihu.com/equation?tex=a%28%5Ccdot%29) ，因为上述的逐顶点运算方式，这两个参数仅与顶点特征相关，与图的结构毫无关系。所以测试任务中改变图的结构，对于GAT影响并不大，只需要改变 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BN%7D_i) ，重新计算即可。与此相反的是，GCN是一种**全图**的计算方式，一次计算就更新全图的节点特征。学习的参数很大程度与图结构相关，这使得GCN在inductive任务上遇到困境。







