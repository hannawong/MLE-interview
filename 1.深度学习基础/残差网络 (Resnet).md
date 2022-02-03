# 残差网络 (Resnet)

#### 1. 问题的提出

网络层数增多一般会伴着下面几个问题:

- 计算资源的消耗。（问题1可以通过GPU集群来解决）
- 模型容易过拟合。 （问题2的过拟合通过采集海量数据，并配合Dropout正则化等方法也可以有效避免）
- 梯度消失/梯度爆炸问题的产生。 （问题3通过正则化初始化和中间的Batch Normalization 层可以有效解决）
  貌似我们只要无脑的增加网络的层数，我们就能从此获益，但实验数据给了我们当头一棒。

作者发现，随着网络层数的增加，**网络发生了退化（degradation）的现象**：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，**训练集loss**反而会增大。注意这并不是过拟合，因为在过拟合中训练loss是一直减小的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417152247627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

#### 2. 残差块的作用

- 只做恒等映射也不该出现退化问题。
  深度网络的退化问题至少说明深度网络不容易训练。但是考虑以下事实：现在已经有了一个浅层神经网络，通过向上堆积新层来建立深层网络。一个极端情况是这些增加的层什么也不学习，仅仅**复制**浅层网络的特征，即向上堆积的层仅仅是在做恒等映射（Identity mapping）。在这种情况下，深层网络应该至少和浅层网络性能一样，也不应该出现退化现象。

- 但是，恒等映射不是想学就能学。

  问题可能是，网络并不是那么容易的就能学到恒等映射。随着网络层数不断加深，求解器不能找到解决途径。
  ResNet 就是通过显式的修改网络结构，加入残差通路，让网络**更容易的学习到恒等映射**。通过改进，我们发现深层神经网络的性能不仅不比浅层神经网络差，还要高出不少。

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417153526871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

  图中的 H(x)代表的是我们最终想要得到的一个映射。在 Plaint net 中，我们就是希望这两层网络能够直接拟合出 H(x)。

  Residual Net：

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417154042616.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



  ### 3. 两种残差学习单元

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417154805463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

  两种结构分别针对ResNet34（左图）和 ResNet50/101/152（右图）。右图的主要目的是减少参数数量。

  为了做个详细的对比，我们这里假设左图的残差单元的输入不是 64-d（通道） 的，而是 256-d 的，那么左图应该为两个 `3×3,256` 的卷积。参数总数为：3 × 3 × 256 × 256 × 2 = 1179648 。对上式做个说明：3×3×256 计算的是每个 filter 的参数数目，第 2 个 256 是说每层有 256 个filter，最后一个 2 是说一共有两层。

  右图的输入同样为 256-d 的，首先通过一个` 1×1,64` 的卷积层将通道数降为 64。然后是一个 `3×3,64` 的卷积层。最后再通过一个 `1×1,256` 的卷积层通道数恢复为 256。参数总数为: 1×1×256×64+3×3×64×64+1×1×64×256=69632。可见参数数量明显变少了。

  通常来说对于常规的ResNet，可以用于34层或者更少的网络中（左图）；对于更深的网络（如101层），则使用右图，其目的是减少计算和参数量。



### 4. 总结：

Resnet的好处主要体现在
[1] 由于直接将原图x经由恒等变换加到卷积之后的F(x)上，给予下一层的模型不同尺度的信息(类似multi-scale GAN)
[2] 增加了skip-connection，能够让梯度直接传播过去而不用经过梯度bottleneck，缓解了梯度消失问题。
[3] 解决了网络层数越多，网络越退化的问题
