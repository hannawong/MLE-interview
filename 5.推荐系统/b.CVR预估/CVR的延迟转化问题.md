## 1. 什么是CVR的延迟转化

### 1.1 计算广告的基础知识

互联网广告的收费模式主要有CPM/CPC/CPA/CPS等模式，不同的计费模式对应广告平台和广告主来说具有不同的利益博弈。比如说，CPM计费（**cost per mille**，千次展示费用），是一种按照**展示即收费**的广告计费模式，而不用管广告的点击率和转化效果。对于平台来说，这样的风险是比较小的，但是广告主的效果就得不到有效的保证了。下图展示了不同的计费方式：

![img](https://pic2.zhimg.com/80/v2-5a3525dd2b21091c04a7f78e062c5505_1440w.jpg)

相对于cpc (cost per click) 计费方式，cpa等方式因为更加注重“转化”，因此更受广告商的欢迎。也就是说，只有当用户注册/订阅之后，广告商才需要付费。

一个重要的公式就是ecpm的计算：ecpm = bid * pctr * pcvr。因此，预测pcvr是十分重要的。

### 1.2 延迟转化问题

什么是延迟转化问题呢？与点击事件不同，广告后续所产生的转化很可能延时发生。比如用户看过一个商品广告，当时有些心动但并没有马上去买，过了几天按捺不住又去购买（Delayed Feedback），给样本打标签带来困难。因为D+1天未购买可能并不一定是真正意义上的未购买，而可能是用户会在**未来**的某一个事件完成购买。这样的标签如果我们直接默认其为负样本，就会有较大的问题 -- 因为它并不是真正意义上的负样本！

例如，在Criteo公司早期，

- 有35%的商品会在点击后一个小时内得到转化；
- 有50%的商品会在点击后24h内得到转化；
- **有13%的商品会在点击后2周之后才得到转化**。

一个最简单的方法是用时间窗口（例如W = 30天），只有看过广告之后的30天内完成转化，才记作正样本；否则为负样本。但是，这种方法对时间窗口的选择带来了挑战：

- 如果时间窗口太短（例如只有两天），一些未来会转化的样本将被错误的标记成负样本；
- 如果时间窗口过长的话，训练集的样本就是W天之前的数据，模型并不是最新的，不适合快速变化的广告以及用户行为模式的变动。例如，Criteo公司早期发现，26天之后就出现了11.3%的新广告。所以，使用30天之前的数据训练出的模型对这部分新的数据预估一定是不准确的。-- 保持模型的**更新**对于广告保持性能至关重要。

## 2. 解决方案

### 2.1 早期的概率方法

CVR的延迟转化问题首次在Criteo的文章*Modeling Delayed Feedback in Display Advertising (KDD'14)* 中得到了比较好的提出和解决。

这篇文章将cvr预估拆分为两个模型，**转化模型（CVR）**和 **延迟模型（DFM）**。CVR模型用户预估用户最终是否发生转化，DFM则预估点击后延迟的时间。两个模型分开建模、联合训练，DFM可以看作一个用来校准(calibrate)CVR的模型。

CVR用logistic regression建模；延迟模型用指数分布建模，即在最后一定会转化的前提下，延迟时间为 ![[公式]](https://www.zhihu.com/equation?tex=d_i) 的概率遵循指数分布：

![img](https://pic1.zhimg.com/80/v2-f55dbd10247fd1912f8eca3f3653b6f0_1440w.jpg)

（回忆一下指数分布...它可以看作几何分布的连续形式：

![img](https://pic1.zhimg.com/80/v2-3de2071f053e4dfea2dc6351b66be6f4_1440w.jpg)指数分布的概率密度函数

根据如上的两个建模，可以得到如下的两个概率（具体推导见原论文）：

![img](https://pic2.zhimg.com/80/v2-091f28bd9afb73c1f823e982ecaa0ae1_1440w.jpg)

在产生训练集的时候，对一个点击后的样本，当跟着一个转化的时候将被标记成正样本，否则将被标记成 unlabeled（这不能直接标记为负样本，因为接下来也可能产生转化）。损失函数如下，就是上面这两个概率的log损失，和极大似然等价：

![img](https://pic4.zhimg.com/80/v2-69e7fe798e3f90fc52f816465c0ec5d3_1440w.jpg)

用梯度下降更新 ![[公式]](https://www.zhihu.com/equation?tex=w_c%2Cw_d) ，即CVR和DFM联合训练。

在线预估时，只使用CVR模型，DFM被舍弃。

### 2.2 京东解决方案

论文：An Attention-based Model for Conversion Rate Prediction with Delayed Feedback via Post-click Calibration (2020)

文中提出CVR预估的两个挑战：

- **数据样本稀少**。对于大量的类别ID型稀疏特征，在CTR预估时往往可以得到不错的embedding，因为CTR使用的数据量很大；但是CVR的数据集是相对较小的。
- 上文所述的**延迟转化**问题，可能带来很多的**假负样本**。因此，直接用标签训练得到的CVR模型是有偏的。

对于数据样本稀少，id类embedding学习不充分的问题，可以使用**CTR**预估时的预训练embedding做初始化。这个思想和ESMM中底层的shared embedding类似（但不是同时训练的），即利用CTR的丰富impression数据学习好的embedding，然后迁移到CVR上。本文则是使用预训练好的Telepath从Item的图像中学习得到结果替换稀疏的ID特征，以此来缓解数据稀少问题。

**转化模型 & 延迟模型**

左侧为转化模型，右侧为时间延迟模型。转化模型是主模型，即预测一个广告在被用户点击之后最后能不能转化；延迟模型是用来对转化模型进行校准(calibrate), 衡量的是**一个尚未转化的样本有多大可能性是真的负样本**。两个模型联合训练，最后只用左侧的转化模型部署上线。

![img](https://pic2.zhimg.com/80/v2-e31188cbe4f8b7caf0f6dd03cbbf2041_1440w.jpg)

**（1）转化模型**

实际上，这个左侧的CVR模型和普通的CTR模型别无二致。 ![[公式]](https://www.zhihu.com/equation?tex=h_c) 是目标商品的稠密特征， ![[公式]](https://www.zhihu.com/equation?tex=u_a) 则是用户历史点击商品信息的加权和 -- 具体地，是用用户历史点击序列先过一层GRU，再过self-attention，然后和目标商品求attention（类似DIN）得到。

![[公式]](https://www.zhihu.com/equation?tex=h_a) 则是对 ![[公式]](https://www.zhihu.com/equation?tex=h_c%2C+u_a) 做了一些人工的交互，最后得到转化的概率：

![img](https://pic2.zhimg.com/80/v2-edb6811b52ad12e7f56c412046219d7d_1440w.jpg)

**（2）延迟模型**

早期的文章都假设，用户在**点击商品之后**到现在的时间是不影响我们商品的最终转化以及转化时间的,但这其实是不对的。很明显地,用户在点击完某个商品之后对于其他商品的点击的确会影响最终是否会购买该商品！所以，这里需要加上特征 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Be_i%7D) , 表示在已过时间[ ![[公式]](https://www.zhihu.com/equation?tex=0%2Ce_i) ]期间新的一些点击。

经过数学推导（参见原文）可以得到：

![img](https://pic1.zhimg.com/80/v2-e72cb8037f76bcaa15fa7c55af575208_1440w.png)

其中，Pr(·)表示时间 ![[公式]](https://www.zhihu.com/equation?tex=d_i) 的延迟概率，这个就是我们要求的东西。它可以由S(·)和h(·)相乘得到。其中，h(·)称为hazard function，是survival analysis中的术语，意为在 ![[公式]](https://www.zhihu.com/equation?tex=d_i) 时存活但之后立即死亡的概率。在这里引申为“在时间 ![[公式]](https://www.zhihu.com/equation?tex=d_i) 前尚未转化、但正好在 ![[公式]](https://www.zhihu.com/equation?tex=d_i) 时转化的概率”。S(·)也是可以由h(·)得出的。所以，其实只需要求h(·)即可。这个延迟模型就是用来求hazard function的。

其中， ![[公式]](https://www.zhihu.com/equation?tex=h_c) 就是candidate item的稠密表征， ![[公式]](https://www.zhihu.com/equation?tex=h_e) 是用户click history的hidden state表示， ![[公式]](https://www.zhihu.com/equation?tex=h_p%28e%29) 则是post-click商品( ![[公式]](https://www.zhihu.com/equation?tex=S_%7Be_i%7D) )经过两层GRU之后的表征。

![img](https://pic1.zhimg.com/80/v2-0a7eb2721bde8e32edff946cb8763d54_1440w.jpg)

**EM训练**

把“最终是否转化”这个事件C当作是未知的**隐藏变量**。

- 在E步，需要计算C的期望，作为C的估计值。很明显，当Y = 1（即已经发生转化）时，必定有C = 1；当Y = 0时，我们需要根据candidate item ![[公式]](https://www.zhihu.com/equation?tex=X_i) , 用户历史行为 ![[公式]](https://www.zhihu.com/equation?tex=H_i) , post-click item ![[公式]](https://www.zhihu.com/equation?tex=S_%7Be_i%7D) 来预估转化的概率。这个概率可以由两个网络的输出得出：

![img](https://pic4.zhimg.com/80/v2-bcf47ddabfced7538d639ea063a1bc5b_1440w.jpg)

在M步，需要利用E步估计的C的期望来**更新网络参数**，具体方法就是使用极大似然估计。具体地，对于那些已经发生转化的商品，似然函数是好求的：

![img](https://pic4.zhimg.com/80/v2-117b8f95d87f91c275a52eab32ac1527_1440w.jpg)

取对数得到log likelihood loss：

![img](https://pic2.zhimg.com/80/v2-b39081f2adc2566841a67ab0d4ca5465_1440w.jpeg)

对于那些尚未发生转化的商品，就需要利用E步求得的 ![[公式]](https://www.zhihu.com/equation?tex=w_i) 了：

![img](https://pic2.zhimg.com/80/v2-4ef6883790acaaa5e688f0dfaabaabe5_1440w.jpg)

取对数得到log-likelihood loss：

![img](https://pic3.zhimg.com/80/v2-0a35a7c0d4eaf947d1c319dbba56b07e_1440w.jpeg)

迭代E、M步，来更新网络参数。

------

参考资料：

[迟到的鸟：cvr 预估中的转化延迟反馈问题概述](https://zhuanlan.zhihu.com/p/74586059)

[DOTA：负样本修正：CVR预估时间延迟问题](https://zhuanlan.zhihu.com/p/353379888)