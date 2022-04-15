# 序列推荐

早期的做法：

- Embedding+MLP：不考虑顺序，只进行简单相加。虽然“看上去”简单相加好像是损失了很多信息，譬如说想象一个二维空间，如果把四个向量位置加权平均，那么最后的点会是这四个点的中点，是个“四不像”。但实际上，在高维空间是我们无法想象的，可以理解为每个点都是一个“小尖尖”上的点，那么平均之后不是“四不像”，而是“四都像”。其实，concat之后+MLP和平均+MLP的表征能力是相似的。见知乎问题：https://www.zhihu.com/question/483946894



## 1. DIN

之前的CTR预估模型大多使用 Embedding+MLP的方法。这样，不管candidate ads是什么，**一个用户只用一个定长的、不变的向量来表示**，这样模型不能充分把握来源于用户丰富历史行为的多样兴趣。为了增强这样一个定长向量的表征能力，需要大大增加这个向量的**维度**。但不幸的是，这样会大大增加参数量、从而导致过拟合。

在这篇[文章](https://arxiv.org/pdf/1706.06978.pdf)中，作者提出Deep Interest Network(DIN)：

1. 使用Attention机制来实现Local Activation unit，能够**针对不同的target ads来计算应该分配给用户历史行为的权重**，从而得到不同的user embedding。
2. 针对模型训练，提出了**Dice**激活函数，**mini-batch aware regularization**，显著提升了模型性能与收敛速度。

#### 1.1 简介

计算广告的一些知识：

- eCPM(effective cost per mille)指的就是每一千次展示可以获得的广告收入, eCPM = bid price * CTR

因此，广告中的CTR预估十分重要。

- QPS (query per second)



#### 1.2 系统总览

在工业界的搜索、广告、推荐架构中，通常包括召回、排序（粗排、精排等）两大模块，无论是对于召回模块还是排序模块来说，对用户历史行为建模都是十分重要的。同时需要指出的是在对用户历史行为进行建模的时候需要了解到用户的行为兴趣是diverse的而且可能是动态变化的，所以如何捕捉用户的多个兴趣点是在用户历史行为建模中非常重要的一点。

#### 1.3 Base模型：Embedding+MLP

Base模型就是现在比较常见的多层神经网络，即先对特征进行Embedding操作，得到一系列Embedding向量之后，将不同group的特征concate起来之后得到一个固定长度的向量，然后将此向量喂给后续的全连接网络。

**1.3.1 Feature Representation**

在CTR预估中的输入特征一般是高维、稀疏的。比如：

![img](https://pic1.zhimg.com/v2-96a108956c7fe6b0ae3d9cf9900913b4_b.jpeg)

其中，visited-cate-ids为用户的历史行为信息，是multi-hot，其他都是one-hot。

论文中作者把特征分为四大类，并没有进行特征交叉，而是**通过DNN去学习特征间的交互信息**(隐式交互)。特征如下：

![img](https://pic3.zhimg.com/v2-2080f6dde8a05dc5eaa5e830e9b0580a_b.jpeg)

**1.3.2 Base Model**

![img](https://pic3.zhimg.com/v2-b253fa951e984c4111071048d420a7ba_b.jpeg)

Embedding：就是去查look-up table。对于one-hot编码，得到一个embedding向量；对于multi-hot编码，得到多个embedding向量，然后做sum-pooling/max-pooling使其降到固定的向量长度。

Concat：把所有的向量（user features, 用户历史行为的pooling,Target ad, Context features）全部concat起来，作为一个综合的表征，送入MLP中。

损失函数为负log损失：

![img](https://pic1.zhimg.com/v2-78460a1b43279de420527aa5209f9864_b.png)

#### 1.4. DIN结构

上文介绍了一些特征，包括user features, user behaviors, Target ad features, Context features.其中，用户历史行为信息(user behaviors)尤其重要。Base模型中，不管candidate ads是什么，一个用户只用一个定长的、不变的向量来表示，不能体现用户多样的兴趣。

我们模型的目标：基于用户历史行为，充分挖掘用户历史行为和候选广告(target ad)之间的关系。用户是否点击某个广告往往是基于他之前的部分兴趣，这是应用Attention机制的基础。例如，一位年轻的母亲购买过一些包包、水杯、童装、鞋子、羽绒服，她的兴趣是多样的。这时，我们要考虑她点击一个羽绒服的概率。其实我们只需要关注她之前买的羽绒服就好了，这个概率和她之前买的包包、水杯等物品关系不大：

> Behaviors related to displayed ad greatly contribute to the click action.

![img](https://pic4.zhimg.com/v2-bb989c51ab7efc9a01643c1e58b93f67_b.png)

DIN结构：

![img](https://pic3.zhimg.com/v2-0100c844be498ba94019c008ac504ff2_b.jpeg)

对历史行为中的每个商品，都增加一个activation unit，计算一下该商品和candidate(target) ad的相关性。activation unit的结构如下：

![img](https://pic1.zhimg.com/v2-0dfde8f6ff6716c7612f0c41dde3cc20_b.jpeg)

```python
def din_attention(self, query, facts, mask, sum = True):

        queries = tf.tile(query, [1, tf.shape(facts)[1]]) ##(?,640)
        queries = tf.reshape(queries, tf.shape(facts)) ##(128,20,32)
        din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)#(128,20,128)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att') #(128,20,1)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]]) #(128,1,20)

        scores = d_layer_3_all

        if mask is not None:
            mask = tf.equal(mask, tf.ones_like(mask))
            key_masks = tf.expand_dims(mask, 1) # [Batchsize, 1, max_len]
            paddings = tf.ones_like(scores) * (-2 ** 32 + 1) # so that will be 0 after softmax
            scores = tf.where(key_masks, scores, paddings)  # if key_masks then scores else paddings

        scores = tf.nn.softmax(scores)  # [B, 1, T]
        if sum:
            output = tf.matmul(scores, facts)  # [B, 1, H]
            output = tf.squeeze(output)
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
            scores = tf.expand_dims(scores,-1)
        return output, scores
```



综合以上，一个用户历史行为的embedding为：

![img](https://pic4.zhimg.com/v2-0bf1bb492f0fa32f2363c6304e27d05b_b.jpeg)

文中作者尝试用LSTM来建模用户行为序列，但没有提升。这是因为用户的兴趣有很多跳跃，带来了噪声。所以，DIN模型**是没有建模序列信息的**（这也正是DIEN的优化点）。



## 2. 训练的Tricks

#### 2.1 Mini-batch Aware Regularization

在大规模稀疏场景下，一些**id类特征的维度很高**，例如在实验中goods_id有6亿维。如果不做正则化，会导致严重的过拟合 (**因为Embedding table的参数量巨大**)，如图中深绿色线所示：

![img](https://pic3.zhimg.com/v2-aa2a7665a0a2dda59c601a8001da6f6a_b.png)

但是，由于Embedding table**参数量巨大**，所以直接使用 L1或者L2 正则化也是不现实的。

> Only parameters of **non-zero** sparse features appearing in each mini-batch needs to be updated in the scenario of SGD based optimization methods without regularization. However, when adding ℓ2 regularization it needs to calculate L2-norm over the **whole** parameters for each mini-batch, which leads to extremely heavy computations and is unacceptable with parameters scaling up to hundreds of millions.

**Mini-Batch Aware regularization** 主要解决的就是在大规模稀疏场景下，采用SGD对引入L2正则的loss进行更新时计算开销过大的问题。因为引入正则化以后，不管特征是不是0，都需要计算梯度，对大规模的稀疏特征，参数规模也非常庞大，增加的计算量就非常大。而MBA方法只对每一个mini-batch中参数不为0的进行梯度更新。

实际上，主要就是Embedding look-up table导致的参数量巨大。Embedding table记作 ![W \in {\mathbb{R} ^{D*K}}](https://www.zhihu.com/equation?tex=W%20%5Cin%20%7B%5Cmathbb%7BR%7D%20%5E%7BD*K%7D%7D),其中K是特征取值的数量，D是embedding vector的维度。![w_j \in \mathbb{R}^{D}](https://www.zhihu.com/equation?tex=w_j%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BD%7D)是W中第 $j$ 个特征对应的embedding。原始的L2正则化项为：

![L_2(W) = ||W||_2^2 = \sum_{j=1}^K||w_j||_2^2](https://www.zhihu.com/equation?tex=L_2(W)%20%3D%20%7C%7CW%7C%7C_2%5E2%20%3D%20%5Csum_%7Bj%3D1%7D%5EK%7C%7Cw_j%7C%7C_2%5E2)

这样的计算量太大了。

推导得到修改过的L2正则项为：

![img](https://pic1.zhimg.com/v2-97277b32d3641bb0966792fb484f7dc4_b.jpeg)

![n_j](https://www.zhihu.com/equation?tex=n_j) 表示特征 ![j](https://www.zhihu.com/equation?tex=j) 在所有样本中出现的次数， ![\alpha_{mj}=1](https://www.zhihu.com/equation?tex=%5Calpha_%7Bmj%7D%3D1)表示当前batch中特征 ![j](https://www.zhihu.com/equation?tex=j) 至少有一个样本非0。这样，**只有在batch m中出现过的特征对应的Embedding table行会计算在L2正则项中**。而且，**正则的强度与特征的频率有关**，频率越高，正则越小，频率越低，正则越大。

求导后得到梯度：

![img](https://pic3.zhimg.com/v2-d4caff9e1d4b2373a7c95c356ce7f166_b.jpeg)



#### **4.2 自适应激活函数Dice**

文章认为采用PRelu激活函数时，它的**rectified point固定为0**，这在每一层的输入分布发生变化时是不适用的。

![img](https://pic4.zhimg.com/v2-3825e57ab73ba7e45776577eb417e84b_b.png)

​                                                                                         (PRelu)

所以文章对该激活函数机型了改进：**平滑**了rectified point附近曲线的同时，激活函数会**根据每层输入数据的分布来自适应调整rectified point的位置**，具体形式如下：

![img](https://pic4.zhimg.com/v2-ca52b032d713301e08cdab7d22b77607_b.png)

其中，

![img](https://pic3.zhimg.com/v2-9913d7a001624f1f653d4f8de79d3756_b.png)

左侧是PRelu的p(s)曲线，右侧是DICE的p(s)曲线



## 2. DIEN -- 加入时序

DIEN实际就是加入了时间序的考虑。这是因为DIN中用户行为序列的条目是被等价对待的，而DIEN考虑了用户兴趣的漂移。多样性->DIN; 进化性->DIEN。

- interest extractor layer: 

​      从用户行为序列中提取时序信息，用GRU来建模时序信息：

![1640401261723](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1640401261723.png)

同时引入了auxiliary loss。这是因为如果只用最后一个hidden state $h_t$ 来进行预测的话，那么只有target loss $L_{target}$ 来**监督最后一个hidden state**，而其它的hidden state并不能够得到有效的监督。所以，用用户在$b_{t+1}$时刻的真实行为$e(t+1)$来监督第$t$个的hidden state $h_t$，同时用负采样来采集负样本$e(t+1)'$。其实，辅助loss也可以看作一种正则化。【正负样本】【辅助loss】

![1640402996877](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1640402996877.png)

auxillary loss 计算公式：

![250fd47a6566453c2638d64787b8226](C:\Users\zh-wa\AppData\Local\Temp\WeChat Files\250fd47a6566453c2638d64787b8226.jpg)

这样，总Loss：

​                                                                            $$L = L_{target} + \alpha * L_{aux}$$ 



- interest evolving layer:【GRU和attention的结合】

![1640404750845](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1640404750845.png)

把给那些和target item相关的sequence子集赋予较高的权重，每个hidden state关于target item的attention为：

![9a6a80000f9e78b2c0d532966977d5e](C:\Users\zh-wa\AppData\Local\Temp\WeChat Files\9a6a80000f9e78b2c0d532966977d5e.jpg)

那么，怎么把GRU和Attention结合在一起呢？直接的想法可能是把hidden state加权，变成$i_t' = h_t *a_t$, 但是，一个全零的向量也会影响GRU的！所以，这种方法不太好。

另一种方法是用attention score来代替GRU中的update gate：$h_t' = (1-a_t)*h_{t-1}^{'}+a_t*h_t^{'}$. 但是，毕竟在原公式中，update gate 是一个向量，乘积也是哈达玛积；如果在这里只用一个标量$a_t$来代替的话，未免太过廉价。所以，在DIEN中实际使用的是AUGRU (GRU with attentional update gate)，用原始的update gate * attention score:

​                                                                                 $\hat{u_t}' = a_t * u_t'$

​                                                                 $h_t' = (1-\hat{u_t'}) \otimes h'_{t-1} + \hat{u_t'} \otimes \hat{h_t'}$

其中 $\otimes$ 表示哈达玛积。





## 4. MIMN - 长序列

随着我们引入更长的历史行为特征之后，会造成系统的**latency和storage cost**增加。所以，类似DIN, DIEN这样的序列推荐方法很难在工业界推行，因为随着行为序列变长，它们的消耗会大幅提升。但是，长序列肯定是可以提升AUC的。那么，怎么在latency和storage cost不增的前提下，尽量使用更长的行为序列呢？

文中作者说，

> Theoretically, the co-design solution of UIC and MIMN enables us to handle the user interest modeling with unlimited length of sequential behavior data. 

但实际上，MIMN只能够解决**千级**的用户行为序列。如何解决更长的用户行为序列，我会在SIM中进行讲解。



#### 4.1 实时CTR预测系统

在线上CTR预估系统中，CTR预估模块接收来自**召回阶段**的候选集之后，会实时的对该候选集中的候选广告进行预测**打分排序**，通常这一过程需要在一定时间内完成，通常是【10毫秒】。

具体线上系统的结构如下图所示：

![img](https://pic3.zhimg.com/80/v2-acb80fffa324ba292e86aebf01977432_1440w.jpg)

左侧为传统的线上实时CTR预估模块，用户行为序列建模是在**predict server**中完成。每次来一个query，都要去分布式数据库中找到用户的行为序列、然后拿到模型中进行建模。这样，用户的大量行为序列都得存储，浪费空间；同时，来了query才去现算，浪费时间。

右侧为基于UIC的线上实时CTR预估模块，将资源消耗最大的**用户兴趣建模功能**单独解耦出来，设计成一个单独的模块**UIC(User Interest Center)**. UIC维护了一个用户最新的兴趣信息，是实时更新的。每次来一个query，都去UIC中直接去查询用户embedding；每次用户有了新的行为，就去更新embedding。

#### 4.2 离线MIMN模型

文章借鉴**神经图灵机（NTM,Neural Turing Machine）**利用额外**存储模块**来解决长序列数据问题的思路，提出了一种全新的CTR预估模型**MIMN**（Multi-Channel User Interest Memory Network），整个系统的模型结构如下图所示：

![img](https://pic1.zhimg.com/80/v2-0a04d679ad5a2d14108541270f470514_1440w.jpg)

左侧：用户行为序列的建模；右侧：传统Embedding+MLP。

##### 4.2.1 神经图灵机 (Neural Turing Machine)

神经图灵机利用一个额外的**记忆网络**来存储长序列信息。在时间t，记忆网络可以表示为矩阵 ![[公式]](https://www.zhihu.com/equation?tex=M_t) ，其包含m个memory slot ![[公式]](https://www.zhihu.com/equation?tex=M_t%28i%29%2C+%5C%7Bi%3D1%2C%E2%80%A6%2Cm%5C%7D) ，NTM通过一个controller模块进行读写操作。这样，用户每来一个行为，就可以更新这个M矩阵，而不用把所有的用户行为序列都存储起来。

![img](https://pic2.zhimg.com/80/v2-8a6eff647a23f16e791a9d4e6f6ba0dd_1440w.jpg)e(1)是第1时刻的用户行为embedding

###### 4.2.1.1 Memory Read

当输入第t个用户行为embedding向量，controller会生成一个用于寻址的read key ![[公式]](https://www.zhihu.com/equation?tex=+k_t) ，首先遍历全部的memory slot，生成一个**权重**向量 ![[公式]](https://www.zhihu.com/equation?tex=w_t%5Er) :

![img](https://pic3.zhimg.com/80/v2-134ba9f3598c3e86d2447fd317510aa6_1440w.jpg)

最后得到一个**加权求和**的结果 ![[公式]](https://www.zhihu.com/equation?tex=r_t) :

![img](https://pic3.zhimg.com/80/v2-5d71d66b84be5d7cc412bf4561cf2e62_1440w.png)

###### 1.4.1.2 Memory Write

类似于memory read中的权重向量的计算，在memory write阶段同样会先计算一个**权重**向量 ![[公式]](https://www.zhihu.com/equation?tex=w_t%5Ew) 。除此之外，还会生成两个向量，一个是add vector ![[公式]](https://www.zhihu.com/equation?tex=a_t) ，另一个是erase vector ![[公式]](https://www.zhihu.com/equation?tex=e_t) ，他们都是controller生成的并且他们控制着记忆网络的更新过程。记忆网络的更新过程如下所示：

![img](https://pic1.zhimg.com/80/v2-decc420ca8727aedd9fef201f7ed9448_1440w.png)

其中，

![[公式]](https://www.zhihu.com/equation?tex=E_t+%3Dw_t%5Ew%5Cotimes+e_t)

![[公式]](https://www.zhihu.com/equation?tex=A_t+%3Dw_t%5Ew%5Cotimes+a_t)

【Memory Utilization Regularization】

传统的NTM存在memory利用分配不平衡的问题，这种问题在用户兴趣建模中尤为重要，因为热门的商品很容易出现在用户行为序列中，这部分商品会主导记忆网络的更新，为此文章提出一种名为memory utilization regularization。该策略的核心思想就是对不同的memory slot的**写权重向量**进行正则，来保证不同memory slot的利用率较为平均。实际上，就是增加了一个reg loss，**为不同slot写权重的方差**。

![img](https://pic3.zhimg.com/80/v2-d107bda2ecaa25daf5d951a46712693a_1440w.jpg)



#### 4.3 memory induction unit：Multi-channel

借鉴NTM的网络结构可以帮助有效构建用户**长**时间序列，但是无法进一步提取用户**高阶**的信息，比如在用户较长的历史浏览行为中，这一系列行为可以被认为是**多个channel**，具体可以表示为如下：

![img](https://pic4.zhimg.com/80/v2-86c08cfc23c89e3439ce288287a7fe5b_1440w.jpg)

因此，文章提出**M**emory **I**nduction **U**nit来对用户**高阶兴趣**进行建模。

![img](https://pic4.zhimg.com/80/v2-ab24110474a474edd52911a582567af3_1440w.jpg)总共有m个GRU链，每个链有T个时间。

在MIU中包含一个额外的存储单元**S**，其包含m个memory slot，这里认为每一个memory slot都是一个用户interest **channel**。在时刻t，MIU首先选择top K的interest channel，然后针对第i个选中的channel，通过下式更新 ![[公式]](https://www.zhihu.com/equation?tex=S_t%28i%29)

![img](https://pic4.zhimg.com/80/v2-37ea5aabe82b969693e758a4eaafbdc7_1440w.png)

![[公式]](https://www.zhihu.com/equation?tex=M_t%28i%29) 对应的是NTM在时刻t第i个memory slot向量， ![[公式]](https://www.zhihu.com/equation?tex=e_t) 为用户行为embedding向量。也就是说**MIU从用户原始输入行为特征和存储在NTM中的信息中提取高阶兴趣信息**。值得一提的是为进一步减少参数量，**不同channel的GRU参数是共享的**.

### 4.4 Implementation for Online Serving

对于整个算法结构来说，在部署到线上的时候，主要的计算量都在网络的左侧部分（即用户行为兴趣建模），而右侧的Embedding+MLP的计算量则小很多。所以在线上部署的时候将左侧的网络部署到UIC server中，将右侧的网络部署到RTP(real-time prediction) server中。

在对左侧用户状态进行更新的时候，最新的memory state代表着用户的兴趣，并且可以存储到TAIR中用户**实时**CTR预估。当有一个用户行为事件发生的时候，UIC会重新计算用户兴趣的表征（而不会将原始的用户行为进行存储）并**增量更新**到TAIR中，所以长时间行为序列占用的系统空间从6T减少到了2.7T。同时文章指出MIMN+UIC这种系统结构不是适用任何场景下的，具体需要符合如下的要求：1、可以得到丰富的用户行为数据；2、用户实时行为的数量不能超过CTR预估的请求数量。