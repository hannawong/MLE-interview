## A) 简单了解BN层 (In 30 seconds)

Batch-Normalization (BN)是一种让神经网络训练**更快**、**更稳定**的方法(faster and more stable)。它计算每个mini-batch的均值和方差，并将其拉回到均值为0方差为1的标准正态分布。BN层通常在nonlinear function的前面/后面使用。

![img](https://pic4.zhimg.com/v2-9e2198bf0ea2a549452cf62a61d17e5f_b.png)

MLP without BN

![img](https://pic4.zhimg.com/v2-d75222a6af3dd250b2bbd5917d0aa6ff_b.png)



## B). BN层的具体计算方法 (In 3 minutes)

##### 0x01. Training 和 Testing 阶段

BN层的计算在training和testing阶段是不一样的。

**在training阶段：**

![img](https://pic3.zhimg.com/v2-4ad9e30dce24ed781f3b37e68a21d36e_b.png)

首先，用(1)(2)式计算一个mini-batch之内的均值 ![\mu_B](https://www.zhihu.com/equation?tex=%5Cmu_B)和方差 ![\sigma_B^2](https://www.zhihu.com/equation?tex=%5Csigma_B%5E2).

然后，用(3)式来进行normalize。这样，每个神经元的output在整个batch上是标准正态分布。

![img](https://pic2.zhimg.com/v2-8d9ade5e79d6d31bd0fbb2d259024279_b.png)

> Example of a 3-neurons hidden layer, with a batch of size b. Each neuron follows a standard normal distribution.

最后，使用可学习的参数 ![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma) 和 ![\beta](https://www.zhihu.com/equation?tex=%5Cbeta)来进行线性变换。这是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入，从而**保证整个network的capacity**没有变小。也可以理解为**找到线性/非线性的平衡**：以Sigmoid函数为例，batchnorm之后数据整体处于函数的非饱和区域，只包含线性变换，破坏了之前学习到的特征分布 。增加![\gamma](https://www.zhihu.com/equation?tex=%5Cgamma) 和 ![\beta](https://www.zhihu.com/equation?tex=%5Cbeta)能够找到一个线性和非线性的较好平衡点，既能享受非线性的较强表达能力的好处，又避免太靠非线性区两头"死区"（如sigmoid）使得网络收敛速度太慢。

**在testing阶段：**

和training阶段不同，在testing阶段，可能输入就只有一个实例，看不到Mini-Batch其它实例，那么这时候怎么对输入做BN呢？为了解决这一问题，我们计算($\mu_{pop}$,$\sigma_{pop}$)，其中：

- $\mu_{pop} $ : estimated mean of the studied population ;
- $σ_{pop}$ : estimated standard-deviation of the studied population.

($\mu_{pop}$ ,$σ_{pop}$)是在训练时计算滑动平均得到的。这两个值代替了在(1)(2)中算出的均值和方差，可以直接带入(3)式。如果阅读了BatchNorm的源码就会发现，这个全局统计量--均值和方差都是通过滑动平均的方法来实现的：

```python
running_mean = momentum * running_mean + (1 - momentum) * x_mean
running_var = momentum * running_var + (1 - momentum) * x_var
```



##### 0x02. 实际应用

Pytorch: [torch.nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), [torch.nn.BatchNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html).

Tensorflow / Keras: [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization), [tf.keras.layers.BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)

在全连接网络中是对**每个神经元**进行归一化，也就是每个神经元都会学习一个γ和β；

在CNN中应用时，需要注意CNN的参数共享机制。每层有多少个**卷积核**，就学习几个γ和β。

##### 0x03. BN层的效果

虽然你可能还不完全理解BN层究竟为什么有用，但是我们都知道，BN层的确有用！

![img](https://pic2.zhimg.com/v2-6d8a2cde69bf184f5fc91fc8974b712d_b.png)

可以看出，BN层能够**让网络更快收敛**、而且**对不同的学习率鲁棒**。在30倍的学习率下，模型依旧能够收敛，而且效果更好（可能因为高学习率帮助模型跳出了local minima）。



## C) 深入理解BN层

> Before BN, we thought that it was almost impossible to efficiently train deep models using sigmoid in the hidden layers. We considered several approaches to tackle training instability, such as looking for better **initialization** methods. Those pieces of solution were heavily heuristic, and way too fragile to be satisfactory. Batch Normalization makes those unstable networks trainable ; that's what this example shows.    — Ian Goodfellow (rephrased from : [source](https://www.youtube.com/watch?v=Xogn6veSyxA))

#### 0x01. BN层的缺点

**Stability Issue**

在testing阶段，我们使用$\mu_{pop},\sigma_{pop}$做为evaluation set的估计值。但是，**假如训练集和测试集分布不同**会怎样呢？比如下图中，训练集是形如左图的sneakers，但是测试集中是形如右图的皮鞋。所以，在测试集中如果还用$\mu_{pop},\sigma_{pop}$, 测试集经过batchnorm之后并不是 ![\mu = 0, \sigma^2 = 1](https://www.zhihu.com/equation?tex=%5Cmu%20%3D%200%2C%20%5Csigma%5E2%20%3D%201)的分布。

![img](https://pic2.zhimg.com/v2-784eac3b291c76b344aca447e9add315_b.png)

> Remark: 训练集和测试集的分布偏差叫做"**covariate shift**".
>
>

#### 0x02.  放在非线性激活函数之前还是之后？

虽然在原论文中，BN层是放在激活函数之前的。因为的确，只有放在激活函数之前才能够起到让输入远离"死区"的效果。但是，后来的实验也表示放在激活函数之后能带来更好的结果。

Keras的创始人之一François Chollet 如是说：

> “I haven't gone back to check what they are suggesting in their original paper, but I can guarantee that recent code written by Christian [Szegedy] applies relu before BN. It is still occasionally a topic of debate, though.” — François Chollet ([source](https://github.com/keras-team/keras/issues/1802))

所以，这个问题没有定论...

#### 0x03. BN层到底为啥有用？

**Hypothesis 1° — BN 降低了 internal covariance shift (ICS) ❌**

[原论文](https://arxiv.org/pdf/1502.03167.pdf) 中提到，Batch Normalization是用来解决"Internal Covariance Shift"问题的。其主要描述的是训练深度网络的时候经常发生训练困难的问题。对于深度学习这种包含很多隐层的网络结构，在训练过程中，**因为各层参数不停在变化**，所以每个隐层都会面临 "covariance shift" 的问题 --- 这就是所谓的"Internal Covariance Shift"，Internal指的是深层网络的隐层，是发生在网络**内部**的事情，而不是covariance shift问题只发生在输入层。

![img](https://pic2.zhimg.com/v2-761a84ce8b93835b35b84cf87d9968bd_b.png)

原论文的逻辑是这样的：

随着网络训练的进行, 每个隐层的参数变化使得后一层的输入发生变化->每批训练数据的分布也随之改变->致使网络在每次迭代中都需要拟合不同的数据分布->增大训练的复杂度以及过拟合的风险。



**Hypothesis 2° — BN降低了每层网络之间的依赖**

![img](https://pic1.zhimg.com/v2-b1b1cae9be606fe9d5dd8b22cb7d6cb4_b.png)

网络层 ![a ](https://www.zhihu.com/equation?tex=a%20)的梯度为：

![img](https://pic1.zhimg.com/v2-dd1f081319c3dfb8e420c4197e2e34e0_b.png)

更新了网络层 ![a](https://www.zhihu.com/equation?tex=a)的参数之后，就会影响后面 ![b,c,d,e](https://www.zhihu.com/equation?tex=b%2Cc%2Cd%2Ce)的输入，这是SGD这样的优化器想不到的，因为优化器只能考虑网络层之间的一阶关系，不能考虑整个网络层序列！

BN层则像一个"阀门"一样控制着水流，每次都标准化、但是还保留了 ![\gamma, \beta](https://www.zhihu.com/equation?tex=%5Cgamma%2C%20%5Cbeta)来做一些变换。

![img](https://pic1.zhimg.com/v2-b381eb1123860ed469091bc778fa686c_b.png)

![img](https://pic1.zhimg.com/v2-d66fb68aacff0f399c2b28cd103a01c8_b.png)


**Hypothesis 3° — BN makes the optimization landscape smoother**

2019年MIT的[一项研究](https://arxiv.org/pdf/1805.11604.pdf) 质疑了一直以来"Internal Covariance Shift"的说法。他们训练在CIFAR-10上训练了三个 VGG 网络，第一个没有任何BN层；第二个有 BN 层；第三个与第二个类似，不同之处在于他们在激活函数之前在中明确添加了一些噪声（随机偏差和方差）。结果如下：

![img](https://pic4.zhimg.com/v2-96b4db36387a8c0f6750e423a60d4f07_b.png)

可以看到，正如预期的那样，第三个网络具有非常高的 ICS。然而，这个"嘈杂"的网络仍然比不含BN层的网络训练得更快，其达到的性能与使用标准 BN 层相当。这个结果表明 BN 的有效性与 ICS 无关! 

这项研究认为，BN效果好是因为BN的存在会**引入mini-batch内其他样本的信息**，就会导致预测一个独立样本时，其他样本信息相当于正则项，使得loss曲面变得更加平滑，更容易找到最优解。相当于一次独立样本预测可以看多个样本，学到的特征泛化性更强，更加general。

通俗来讲， 不进行BN， loss不仅仅非凸且趋向于坑洼，平坦区域和极小值，这使得优化算法极不稳定，使得模型对**学习率**的选择和**初始化方式**极为敏感，而BN大大减少了这几种情况发生。

> BN relies on batch first and second statistical moments (mean and variance) to normalize hidden layers activations. The output values are then strongly tied to the current batch statistics. Such transformation adds some noise, depending on the input examples used in the current batch. Adding some noise to avoid overfitting … sounds like a regularization process, doesn't it ? :)

![img](https://pic3.zhimg.com/v2-723144762c91a6b5e7b24eb29eee56ba_b.png)



**Hypothesis 4: 防止梯度消失**

这很好理解， BN 将激活函数的输入数据压缩在 N(0,1) 空间内，的确能够很大程度上减轻梯度消失问题。









https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338#b93ctowardsdatascience.com




  