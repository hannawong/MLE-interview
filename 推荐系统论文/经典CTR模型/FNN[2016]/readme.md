# Factorisation-machine supported Neural Networks (FNN)——求援于FM

> FM的精髓，最上在于latent embedding，有了它才能把交互**拆解**到基底上；居中在于element-wise乘，能让两个特征之间**互相影响**；最下在于**点积**，把好不容易带进来的高维信息全部压缩完了



深度神经网络在CV，NLP上已经取得巨大成功，开始延伸到推荐领域的时候。学者们首先要考虑的问题是，如何把一个高度稀疏，维度也很高的拼接输入放到DNN中去。这里要特别说明，在当时学术界的背景下，特征往往是one-hot并且可以穷举的。比如一个特征是城市，一共有1000中选择，就有一个1000维的向量，其中只有一个地方有非零值。当我把所有这类特征拼接起来的时候就是一个非常高维的输入了。后面会讲现代的embedding生成方式，DNN的输入会变成比较短的dense输入，拼接起来不会特别高维。这种特征也可以称之为categorical的特征。

FNN就是其中一个工作，他的想法是，**把FM当做一个很好的初始化工具**。从serving的角度来看，其本质是一个很简单的MLP：

![img](https://pic3.zhimg.com/80/v2-dd02c9900bb39dd3ef4018cf383ac86e_1440w.jpg)

在上图中分了两个field，但为了便于理解可以先假设只有一个。假设有 ![[公式]](https://www.zhihu.com/equation?tex=N) 种feature (也就是one-hot拼起来是 ![[公式]](https://www.zhihu.com/equation?tex=N) 维），一开始就就有一个从 ![[公式]](https://www.zhihu.com/equation?tex=x) 到 ![[公式]](https://www.zhihu.com/equation?tex=z) 的权重矩阵，接下来的几层就是很普通的MLP。在serving的时候就是很普通的categorical特征输入DNN的结构。

但在训练时较为特殊：我们先训练一个FM，就可以得到每一种特征正对应的一个 ![[公式]](https://www.zhihu.com/equation?tex=v_i) 。**接下来用这些embedding来初始化从one-hot输入到第一层dense激活元之间的那个权重矩阵** ![[公式]](https://www.zhihu.com/equation?tex=W) 。当然也要注意，**FNN的训练是要分两阶段的，FM阶段和DNN阶段**。

FM对于FNN是一种援助的关系，仅仅出现在**初始化**的地方，或者说是给出了一种DNN的优化方式。但是这种方法不是端到端的，不利于online-learning。而且，FNN只考虑了高阶特征交叉，没有保留低阶的特征。