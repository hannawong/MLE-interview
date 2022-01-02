# SimCSE

Simple Contrastive Learning of Sentence Embeddings，其实就是把对比学习引入了SBERT，达到了**句子相似度**SOTA。SBERT本身并不复杂，仅仅是一个基于BERT的孪生网络而已，想要在SBERT的基础上改进指标，只能从训练目标下手。

### 1. 对比学习概念

对比学习的思想很简单，即拉近相似的样本，推开不相似的样本，一种常用的对比损失是基于mini-batch采样负样本的交叉熵损失，假设我们有一个数据集![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BD%7D%3D%5Cleft%5C%7B%5Cleft%28x_%7Bi%7D%2C+x_%7Bi%7D%5E%7B%2B%7D%5Cright%29%5Cright%5C%7D_%7Bi%3D1%7D%5E%7Bm%7D)，其中![[公式]](https://www.zhihu.com/equation?tex=x_i)和![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)是语义相关的，则在大小为![[公式]](https://www.zhihu.com/equation?tex=N)的mini batch内，![[公式]](https://www.zhihu.com/equation?tex=%28x_i%2C+x_%7Bi%7D%5E%7B%2B%7D%29)的训练目标为

![[公式]](https://www.zhihu.com/equation?tex=%5Cell_%7Bi%7D%3D%5Clog+%5Cfrac%7Be%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%2C+%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7B%2B%7D%5Cright%29+%2F+%5Ctau%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BN%7D+e%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%2C+%5Cmathbf%7Bh%7D_%7Bj%7D%5E%7B%2B%7D%5Cright%29+%2F+%5Ctau%7D%7D+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cdisplaystyle%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7B1%7D%2C+%5Cmathbf%7Bh%7D_%7B2%7D%5Cright%29%3D%5Cfrac%7B%5Cmathbf%7Bh%7D_%7B1%7D%5E%7B%5Ctop%7D+%5Cmathbf%7Bh%7D_%7B2%7D%7D%7B%5Cleft%5C%7C%5Cmathbf%7Bh%7D_%7B1%7D%5Cright%5C%7C+%5Ccdot%5Cleft%5C%7C%5Cmathbf%7Bh%7D_%7B2%7D%5Cright%5C%7C%7D)，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D_i)和![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7B%2B%7D)是![[公式]](https://www.zhihu.com/equation?tex=x_i)和![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)的编码表示，![[公式]](https://www.zhihu.com/equation?tex=%5Ctau)为softmax的温度超参。

分子是真正的正样本，分母是正样本+所有负样本，这个其实就是个交叉熵损失。

#### 1.1 怎么构造正样本

使用对比损失最关键的问题是如何构造![[公式]](https://www.zhihu.com/equation?tex=%28x_i%2C+x_%7Bi%7D%5E%7B%2B%7D%29)，对比学习最早起源于CV领域的原因之一就是图像的![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)非常容易构造，**裁剪、翻转、扭曲和旋转**都不影响人类对图像语义的理解，因此可以直接作为正样本。而结构高度**离散**的自然语言则很难构造语义一致的![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)，前人采用了一些数据增强方法来构造![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)，比如**替换、删除、重排**，但这些方法都是离散的操作，很难把控，容易引入负面噪声，模型也很难通过对比学习的方式从这样的样本中捕捉到语义信息，性能提升有限。



#### 1.2 句子embedding的好坏评判标准：Alignment & uniformity

- alignment计算![[公式]](https://www.zhihu.com/equation?tex=x_i)和![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)的平均距离：

![[公式]](https://www.zhihu.com/equation?tex=%5Cell_%7B%5Ctext+%7Balign+%7D%7D+%5Ctriangleq+%5Cunderset%7B%5Cleft%28x%2C+x%5E%7B%2B%7D%5Cright%29+%5Csim+p_%7B%5Ctext+%7Bpos+%7D%7D%7D%7B%5Cmathbb%7BE%7D%7D%5Cleft%5C%7Cf%28x%29-f%5Cleft%28x%5E%7B%2B%7D%5Cright%29%5Cright%5C%7C%5E%7B2%7D+%5C%5C)

自然是希望正样本和正样本的距离越近越好。

- uniformity计算向量整体分布的均匀程度：


![[公式]](https://www.zhihu.com/equation?tex=%5Cell_%7B%5Ctext+%7Buniform+%7D%7D+%5Ctriangleq+%5Clog+%5Cunderset%7Bx%2C+y%5Csim+p_%7B%5Ctext%7Bdata%7D%7D%7D%7B%5Cmathbb%7BE%7D%7D+e%5E%7B-2%5C%7Cf%28x%29-f%28y%29%5C%7C%5E%7B2%7D%7D+%5C%5C)

![img](https://pic1.zhimg.com/80/v2-00663c0ed44da5eeac9e28d741d0a95c_1440w.jpg)

我们希望语义向量要尽可能地**均匀分布在超球面上**，因为均匀分布**信息熵最高**，分布越均匀则保留的信息越多。

“拉近正样本，推开负样本”实际上就是在优化这两个指标。



## 2. SimCSE

#### 2.1 无监督的SimCSE

本文作者提出可以通过dropout 来生成正样本![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5E%7B%2B%7D)，回想一下，在标准的Transformer中，dropout mask被放置在**全连接层和注意力**操作上。由于dropout mask是随机生成的，所以在训练阶段，将同一个样本分两次输入到同一个编码器中，我们会得到两个不同的表示向量![[公式]](https://www.zhihu.com/equation?tex=z%2Cz%5E%5Cprime)，将![[公式]](https://www.zhihu.com/equation?tex=z%5E%5Cprime)作为正样本，则模型的训练目标为

![[公式]](https://www.zhihu.com/equation?tex=%5Cell_%7Bi%7D%3D-%5Clog+%5Cfrac%7Be%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7Bz_%7Bi%7D%7D%2C+%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7Bz_%7Bi%7D%5E%7B%5Cprime%7D%7D%5Cright%29+%2F+%5Ctau%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BN%7D+e%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7Bz_%7Bi%7D%7D%2C+%5Cmathbf%7Bh%7D_%7Bj%7D%5E%7B%7Bz_%7Bj%7D%7D%5E%7B%5Cprime%7D%7D%5Cright%29+%2F+%5Ctau%7D%7D+%5C%5C)

这种通过改变dropout mask生成正样本的方法可以看作是**数据增强**的最小形式，因为原样本和生成的正样本的语义是完全一致的，只是生成的embedding不同而已。所以，其实SimCSE生成正样本的方式就是把样本过两次预训练好的BERT，用dropout来获得两个不一样的embedding作为正例对；负样本做mini-batch采样....对，就这。

#### 2.2 有监督的SimCSE

在SBERT原文中，作者将NLI数据集作为一个**三分类**任务来训练(entailment, neutral, contradiction)，这种方式忽略了正样本与负样本之间的交互，而**对比损失**则可以让模型学习到更丰富的细粒度语义信息。

构造训练目标其实很简单，直接将数据集中的正负样本拿过来用就可以了，将NLI数据集中的entailment作为正样本，contradiction作为负样本，加上原样本premise一起组合为![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28x_%7Bi%7D%2C+x_%7Bi%7D%5E%7B%2B%7D%2C+x_%7Bi%7D%5E%7B-%7D%5Cright%29)，并将损失函数改进为

![[公式]](https://www.zhihu.com/equation?tex=-%5Clog+%5Cfrac%7Be%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%2C+%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7B%2B%7D%5Cright%29+%2F+%5Ctau%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BN%7D%5Cleft%28e%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%2C+%5Cmathbf%7Bh%7D_%7Bj%7D%5E%7B%2B%7D%5Cright%29+%2F+%5Ctau%7D%2Be%5E%7B%5Coperatorname%7Bsim%7D%5Cleft%28%5Cmathbf%7Bh%7D_%7Bi%7D%2C+%5Cmathbf%7Bh%7D_%7Bj%7D%5E%7B-%7D%5Cright%29+%2F+%5Ctau%7D%5Cright%29%7D+%5C%5C)