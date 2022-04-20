# DIEN -- 加入时序

DIEN实际就是加入了时间序的考虑。这是因为DIN中用户行为序列的条目是被等价对待的，而DIEN考虑了用户兴趣的漂移。**多样性->DIN; 进化性->DIEN**。

## 1. 模型结构

![img](https://pic2.zhimg.com/80/v2-debfeab0c062f8473f53ee9635c5f4dc_1440w.png)

### 1.1 interest extractor layer

图中浅黄色部分，即从用户行为序列中提取时序信息，用GRU来建模即可。GRU输出的hidden state为 {h(1), h(2),...h(T)}

这里还有一个比较新颖的点是引入了auxiliary loss。这是因为如果只用**最后一个hidden state** $h_t$ 来进行预测的话，那么只有target loss $L_{target}$ 来监督最后一个hidden state，而其它的hidden state并不能够得到有效的监督。这个loss传回去的时候不免会有**梯度消失**，导致前面的hidden state不能有效更新。所以，用用户在t+1时刻的真实行为e(t+1)$来监督第$t个的hidden state $h_t$，同时用负采样来采集负样本$e(t+1)'$，这样能够增强hidden state 的**expressiveness**。其实，辅助loss也可以看作一种正则化。【正负样本】【辅助loss】

![1640402996877](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1640402996877.png)

auxillary loss 计算公式：

![img](https://pic3.zhimg.com/80/v2-db87b218194122b4c924a7c6caa58471_1440w.png)

意思是，第t步的hidden state $h_t$ 应该和第t+1步的真实click样本embedding离得越近，和t+1步的负样本离得越远。

这样，总Loss：

​                                                                            $$L = L_{target} + \alpha * L_{aux}$$ 

这样，第一步GRU输出的hidden states就能够把握住足够多的信息，是比较好的user sequence 表征了。它将作为下一步Interest evolving layer的输入。

### 1.2 interest evolving: GRU 和 Attention的结合

和DIN的思想一样，我们希望赋予那些和target item更相关的行为序列子集以较高的权重，让它们自成一个GRU兴趣序列，防止其他不相关的兴趣对这个target item的预估产生干扰。

每个hidden state $h_t$ 关于target item $e_a$ 的attention score计算公式如下：



那么，怎么把GRU和Attention结合在一起呢？直接的想法可能是把hidden state加权，变成$i_t' = h_t *a_t$, 但是，一个全零的向量也会影响GRU的！（even zero input can also change the hidden state of GRU）所以，这种方法不太好。

另一种方法是用attention score来代替GRU中的**update gate**：$\mathbf{h_t'} = (1-a_t)*\mathbf{h_{t-1}^{'}}+a_t*\mathbf{\tilde{h_t^{'}}}$. 当注意力权重接近于1时，我们会分配给当前的更新向量 $\mathbf{\tilde{h_t^{'}}}$ 以很大的权重，这就达到了给GRU融合attention权重的效果！

但是，毕竟在原公式中，update gate 是一个向量，乘积也是哈达玛积；如果在这里只用一个标量$a_t$来代替的话，未免太过廉价。

所以，在DIEN中实际使用的是AUGRU (GRU with attentional update gate)，**用原始的update gate * attention** score:

​                                                                                 $\mathbf{\hat{u_t}'} = a_t * \mathbf{u_t'}$

​                                                                 $\mathbf{h_t'} = (1-\mathbf{\hat{u_t'}}) \otimes \mathbf{h'_{t-1}} + \mathbf{\hat{u_t'}} \otimes \mathbf{\hat{h_t'}}$

其中 $\otimes$ 表示哈达玛积。

----

工业数据集的构建：使用曝光/点击日志，用用户最近49天点击的商品作为target item，对于每个target item都构建之前的用户行为序列（向前推14天）。