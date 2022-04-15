# DIN

早期的做法：

- Embedding+MLP：不考虑顺序，只进行简单相加。虽然“看上去”简单相加好像是损失了很多信息，譬如说想象一个二维空间，如果把四个向量位置加权平均，那么最后的点会是这四个点的中点，是个“四不像”。但实际上，在高维空间是我们无法想象的，可以理解为每个点都是一个“小尖尖”上的点，那么平均之后不是“四不像”，而是“四都像”。其实，concat之后+MLP和平均+MLP的表征能力是相似的。见知乎问题：https://www.zhihu.com/question/483946894



## 1. DIN

之前的CTR预估模型大多使用 Embedding+MLP的方法。这样，不管candidate ads是什么，**一个用户只用一个定长的、不变的向量来表示**，这样模型不能充分把握来源于用户**丰富历史行为**的**多样兴趣(diverse interest)**，所以这个定长的向量就会称为表征的bottleneck。为了增强这样一个定长向量的**表征能力**，需要大大增加这个向量的**维度**。但不幸的是，这样会大大增加参数量、从而导致过拟合。

在这篇[文章](https://arxiv.org/pdf/1706.06978.pdf)中，作者提出Deep Interest Network(DIN)：

1. 使用Attention机制来实现Local Activation unit，能够**针对不同的target ads来计算应该分配给用户历史行为的权重**，从而得到不同的user embedding，这样能够打破定长向量这个bottleneck，增强模型的表征能力。
2. 针对模型训练，提出了**Dice**激活函数，**mini-batch aware regularization**，显著提升了模型性能与收敛速度。

#### 1.1 简介

计算广告的一些知识：

- eCPM(effective cost per mille)指的就是**每一千次展示可以获得的广告收入**, eCPM = bid price * CTR

因此，广告中的CTR预估十分重要。

- QPS (query per second)



#### 1.2 系统总览

在工业界的搜索、广告、推荐架构中，通常包括召回、排序（粗排、精排等）两大模块，无论是对于召回模块还是排序模块来说，对用户历史行为建模都是十分重要的。同时需要指出的是在对用户历史行为进行建模的时候需要了解到用户的行为兴趣是diverse的而且可能是动态变化的，所以如何捕捉用户的多个兴趣点是在用户历史行为建模中非常重要的一点。

#### 1.3 Base模型：Embedding+MLP

Base模型就是现在比较常见的多层神经网络，即先对特征进行Embedding操作，得到一系列Embedding向量之后，将不同group的特征concate起来之后得到一个固定长度的向量，然后将此向量喂给后续的全连接网络。这样的方法比起之前的logistic regression，不用手动构造交叉特征，MLP完成隐式的特征交叉。

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

上文介绍了一些特征，包括user features, user behaviors, Target ad features, Context features.其中，用户历史行为信息(user behaviors)尤其重要。Base模型中，不管candidate ads是什么，一个用户只用一个定长的、不变的向量来表示，不能体现用户【多样的兴趣】。

我们模型的目标：基于用户历史行为，充分挖掘用户历史行为和候选广告(target ad)之间的关系。用户是否点击某个广告往往是基于他之前的**部分兴趣**，这是应用Attention机制的基础，所以我们应该对历史行为进行soft search，只去关注那些和target ads有关的行为，给他们以高的权重，最后的行为向量表示就是一个加权求和。例如，一位年轻的母亲购买过一些包包、水杯、童装、鞋子、羽绒服，她的兴趣是多样的。这时，我们要考虑她点击一个羽绒服的概率。其实我们只需要关注她之前买的羽绒服就好了，这个概率和她之前买的包包、水杯等物品关系不大：

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

“only update those parameters of sparse features **appearing** in each mini-batch”.

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

"adaptively adjust the rectified point w.r.t. distribution of inputs"

![img](https://pic4.zhimg.com/v2-ca52b032d713301e08cdab7d22b77607_b.png)

其中，

![img](https://pic3.zhimg.com/v2-9913d7a001624f1f653d4f8de79d3756_b.png)

左侧是PRelu的p(s)曲线，右侧是DICE的p(s)曲线