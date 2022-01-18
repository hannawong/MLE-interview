# Latent Dirchlet Allocation (LDA 主题模型)

LDA (2003年) 解决的问题是，给你一个document，怎么去得到它的主题分布，例如[科学:0.8, 政治:0.1, 体育:0.1]。当然，主题的个数是人工选定的一个超参数。此模型可以用来做文本聚类（无监督学习）。

LDA把文档当成一个**词袋**，即不考虑文档中出现token的顺序。



### 1. LDA是啥

把LDA想成是一个产生fake文档的机器。每个机器都不同的setting，从而产生不同的fake文档。总有一个LDA机器能够产生最接近于原文的词语，那么这个机器的setting就是我们要找的主题分布。

LDA机器长这样：

![img](https://pic3.zhimg.com/80/v2-687de084b281d134539e8a73b4c60856_1440w.png)

它的输入是两个狄利克雷分布alpha、beta, 输出一堆words（词袋）。具体的公式如下：

![img](https://pica.zhimg.com/80/v2-f6647898b4719ed109211c3e5757923c_1440w.png)





### 2. 狄利克雷分布

想象一下我们有一个三角形的房间，在举办一个party，人们本来是在屋子中均匀分布的.（图1）

我们在三个顶点上放上美酒、音乐和美食，那么人们自然会靠近这三个顶点来分布，分布在中间的概率是较小的。（图2）

如果我们在三个顶点上放上大火、狮子和核辐射，那么人们肯定会向中间聚拢。（图3）

![img](https://pic2.zhimg.com/80/v2-45199cd90aa697ab00fab4c59b2e10e0_1440w.png)

狄利克雷分布的公式为：

![img](https://pic1.zhimg.com/80/v2-df2f81dc5d1967cd16ea473b52cdee9c_1440w.png)

这里的alpha是一个向量，即每个顶点的alpha值。

每个document的主题狄利克雷分布长这样：

![img](https://pic3.zhimg.com/80/v2-747f5b31b743d9e5139e4217d884c735_1440w.png)

现在我们只有三个topic，那么如果有四个、五个...topic会怎么样呢？实际上，更多的topic就需要更多的维度。例如，如果只有两个topic，就是一条直线；三个topic是个三角形；四个topic就是一个三棱锥（为了保证每个顶点之间的距离一样）。topic的个数是一个需要人来手动调整的超参数。

![img](https://pica.zhimg.com/80/v2-762678eed7d6d7f2f4dbac0e392d7bdd_1440w.png)

现在，我们把词语放在三棱锥的顶点上，那么topic的分布是什么样子的呢？

![img](https://pic3.zhimg.com/80/v2-3edb248ecd3b4f30ad5ff1bf39e1872f_1440w.png)

现在，我们有了document-topic和topic-word狄利克雷分布。

那么，怎么根据这两个狄利克雷分布来生成document呢？

![img](https://pica.zhimg.com/80/v2-841aa212d348103a16bdd22d00edcb5c_1440w.png)

顺序是这样的：首先我们有了第一个document-topic狄利克雷分布，即我们要求的每个document的主题分布；这个主题分布对应第三个多项式分布，即我们有70%的概率选择science主题、20%的概率选择sports主题、10%的概率选择politics主题。我们还有第二个topic-word狄利克雷分布，即对于每个主题，我们选择不同词语的概率都是不同的（就像一个作家针对不同topic遣词造句的时候会根据不同主题选择不同的词）。那么怎么构造一个fake document呢？其实就是先根据一定概率选topic（第一个狄利克雷分布），再对不同的topic按照一定的概率选词语（第二个狄利克雷分布）。这样，我们最后得到了一个fake document。

我们的目的就是要求最好的两个狄利克雷分布。为了选择最好的，我们需要尝试不同的分布，让它们生成fake document（使用泊松分布来生成document的长度），最接近于实际的document的那个分布就是最好的分布。

### 3. 吉布斯采样

但是，所有可能的两个狄利克雷分布是无穷多的，我们毕竟不能每个都尝试。那么，该怎么训练一个好的LDA呢？

![img](https://pic2.zhimg.com/80/v2-ff1b2ef80be8a9d2be1b9f7f7f46a609_1440w.png)

我们的目的：

- 每个document尽量都染成一个颜色；
- 每个word都尽量染成一个颜色

为了达到这个目的，就需要吉布斯采样了。吉布斯采样就像整理一个混乱的房间。我们现在不知道每个家具都应该放在哪里，但是我们知道每件家具的**相对位置**，例如：电脑应该放在桌子上、衣服应该放在衣柜里。那么，在整理屋子的时候，我们每次都**假设其他家具已经放在了正确位置**，然后对于一个随机选择的家具，我们都把它放在正确的位置...知道所有物品都不会再移动。（其实这也是贪心的思想？）

我们使用吉布斯采样来对word进行染色。

- 首先，随机对每个document中的word进行染色（类比混乱的房间）
- 然后，每次都随机选一个word，我们假设其他所有的word都已经被最佳染色，然后现在要给当前的word染最好的颜色，从而达到”每个document尽量都染成一个颜色；每个word都尽量染成一个颜色“的效果。

![img](https://pic1.zhimg.com/80/v2-e02919a39e9941558ae3893f66fc3d72_1440w.png)

​     为了找到这个颜色，计算一个乘积：

![img](https://pica.zhimg.com/80/v2-f5a9aaa43e426bfd76aec102d251c4d4_1440w.png)

当然了，我们并不是取乘积最大的那个颜色，而是将乘积作为概率随机选择的。乘积最大的颜色有更高的概率被选中：

![img](https://pic3.zhimg.com/80/v2-3c2c39a5ffb5c5b252f8833d43e4733b_1440w.png)

- 对每个word都染色之后，便可以得到每个document中topic的”含量“：

![img](https://pic3.zhimg.com/80/v2-8afe2895662eba91066c08855874934a_1440w.png)

当然了，计算机只知道聚类(topic1,topic2,topic3)，但是并不知道每个聚类的含义。我们需要人工给每个topic以意义，例如science、politic、sports。