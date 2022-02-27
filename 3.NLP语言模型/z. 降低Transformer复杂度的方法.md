# 降低Transformer复杂度的方法

### 1. Sparse Transformer

Sparse Attention是为了解决Transformer模型随着长度n的增加，Attention部分所占用的内存和计算呈平方增加的问题。回忆Transformer的复杂度为$O(n^2d)$, 而sparse transformer试图把此复杂度降低为$O(n\sqrt{n}d)$.这样，就可以处理上千长度的输入，层数可以达到上百层。

#### 1.1 Intuition

Transformer的Decoder部分是一个**自回归（AR）**模型。对于图像生成任务，可以把图像的像素点按照从上到下从左到右的方式当成一个序列，然后在序列上去做自回归。

论文中首先构造了一个128层的full-attention网络，并在Cifar10图像生成问题上进行了训练。如下图所示，底部的黑色部分表示尚未生成到的部分，白色凸显的部分则是当前步注意力权重高的地方。

![img](https://pic1.zhimg.com/80/v2-d25545e2853a472df737c901260181f6_1440w.png)

(a)中是比较低层的layer的注意力，可以看到，低层次的时候主要关注的还是**局部区域**的部分。

(b)在第19层和20层，Attention学习到了横向和纵向的规律。

(c)还有可能学习到和数据本身相关的attention。比如下图，第二列第二张学习到了鸟的边缘。

(d) 64-128层的注意力是高度稀疏的，只有极少的像素点有较高的注意力。

无论如何，注意力权重高的地方只占一小部分，这就为**稀疏注意力**提供了数据上的支持。作为解决注意力平方问题的早期论文，本文从图像生成的问题上揭示了attention的原罪，那就是其实不需要那么**密集**的注意力，Top-k的注意力已经足够可以保证效果了。

#### 1.2 Factorized Self-attention

Sparse Transformer就是把full self-attention 分解成若干个小的、复杂度低的self-attention。这个过程叫做factorization。

定义集合$S = {S_1,...S_n}$， 这个集合中的每个元素还是集合，表示第i位input可以关注的位置。对于full-attention，$S_i$ 显然就是 {j:j<i}. 每个位置的attention现在就变成了下图公式。其实没多大变化，只不过以前可以关注自己之前的所有位置，现在只能关注到一些特定的位置而已。

![img](https://pic3.zhimg.com/80/v2-a7a1c8ed658b9ba5f22dab2ed762b68d_1440w.jpeg)

对于factorized self-attention，使用p个sparse注意力头，每个sparse注意力头有着不同的关注列表，记作$A_i^{(m)}$.

- 为了保证sparse注意力头的高效性(**efficiency**), 我们必须要保证 $|A_i^{(m)}|$是 $O(^p\sqrt{n})$复杂度的。
- 同时，为了保证sparse注意力头是有效(valid)的，我们需要保证每个位置都可以经过一些路径attend到**之前所有位置**（毕竟，这样才属于"factorize" full-attention）。同时这个路径长度不超过p+1，这样保证所有原本在全注意力上能够传递的信号在稀疏注意力的框架下仍然可以有效传递。

##### 1.2.1 两种可能的sparse attention方法

当p = 2时，即两个注意力头的时候，文章给出了如下两种可以的sparse attention方法，能够满足上文所述的efficiency和valid条件。

（1）strided attention

- 一个注意力头只能关注当前位置前 $l = \sqrt{n}$ 个位置
- 另一个注意力头只能关注当前位置前面隔 $l = \sqrt{n} $ "跳"的位置

![img](https://pic3.zhimg.com/80/v2-133b492e8c7dee8165e11e65cf563660_1440w.jpeg)

这样相当于关注当前行、当前列的信息，就如之前看的图像生成例子中的（b）一样。所以，这种注意力机制比较适用于图像。

（2）fixed attention

- Ai(1) = {j: floor(j/l) = floor(i/l)}
- Ai(2) = {j: j mod l ∈ {t, t+1, ..., l}}，其中t=l-c且c是超参数。

一般情况下，l取值为{128, 256}, c取值为{8, 16, 32}。



**稀疏注意力的组合**

一个直接的方法是在每个层使用同样的稀疏机制，在不同的块使用不同的。这样每个层的不同机制”交织(interleave)“在一起。

另一种方式则是在每个层使用组合的稀疏注意力，组合的方法则是把经过不同稀疏注意力机制的输出concat起来，就像普通的多头一样。

**深度残差Transformer**

深层次的Transformer训练起来十分困难，因为使用残差的方式会比较好。除了我们熟悉的transformer层内的layernorm之外，还增加了层间的残差连接，可以处理上百层的层。

![img](https://pic1.zhimg.com/80/v2-0454e3af14036b81e4a65e4eaf3bbdfc_1440w.jpeg)

![img](https://pic1.zhimg.com/80/v2-e71dbb524c754504b02ddfb76908f81e_1440w.jpeg)







