集成学习



## 0x01. 简介

#### 1. 使用集成学习的原因

reason 1:

If the training examples are few and the hypothesis space is large then there are several equally accurate classifiers. Selecting one of them may lead to bad result on the test set.

![img](https://pic4.zhimg.com/v2-e673a9dbe1bb901fbfc58f35ac68954f_b.png)

reason 2:

Algorithms may converge to local minima. Combining different hypotheses may lower the risk of bad local minimum.

![img](https://pic1.zhimg.com/v2-abf62dcd13fc519fdfb0e2fbaff60900_b.png)

reason 3:

Hypothesis space determined by the current algorithm does not contain the true function, but it has several good approximations.

![img](https://pic3.zhimg.com/v2-13271e4b129d63c8f3eb9df7f7a05556_b.png)

2. **好的弱分类器的特征**

![img](https://pic3.zhimg.com/v2-72ed39a9fd10c5ed9f7f23dbda8b6fc6_b.png)

为了在集成时达到比较好的效果，每个弱分类器需要good且different.



### 0x02. 加权多数算法(weighted majority algorithm)

![img](https://pic3.zhimg.com/v2-d298f53c7f86202fff329c15356f95f2_b.png)

对每个弱判别器的输出结果加一个权重，最后的输出结果是所有弱判别器结果的加权平均。



### 0x03. Bagging(Boot-strap Aggregating,拔靴法提升)

如果我们只有**一个弱分类器模型**，该怎么集成才能提升弱分类器的效果呢？

#### 1. Intuition 

同一个弱分类器在不同的数据上训练，会训出来不同的模型。那么，可以从数据集里采样采得不同的训练集，然后训练不同的模型。这些模型会是不同的，但是他们的效果可能很差。

> A naïve approach: sample different subsets from the training set and train different base models - These models should be quite different - But their performances may be very bad

#### 2. Bootstrap Sampling(拔靴法采样)

- 数据集D含有m个数据点
- 从D中有放回得取出m个数据点，构成了 ![D_1](https://www.zhihu.com/equation?tex=D_1) 数据集
- ![D_1](https://www.zhihu.com/equation?tex=D_1) 数据集就是D的一个子集

#### 3. Bagging算法

![img](https://pic1.zhimg.com/v2-078a8bcc3bd4c5df5d00fb541780016c_b.png)

#### 4. 测试结果

![img](https://pic2.zhimg.com/v2-aafc1bab2f8d0303309d460b387110c9_b.png)

   $e_S$是决策树的error,$e_B$是bagging之后的error，发现bagging之后有了明显效果提升

![img](https://pic2.zhimg.com/v2-894c56f995d3d709c908702429d50e69_b.png)

 $e_S$是KNN的error,$e_B$是bagging之后的error，发现bagging之后竟然没有变化！

bagging可以显著提升决策树的效果，但是不能提升KNN的效果的原因是：

- Bagging只能帮助那些unstable的模型("The vital element is the instability of the prediction method",e.g. Decision tree, neural network)
- Unstable: 训练集改变一点点，会导致模型的巨大变化. 

> "If perturbing the learning set can cause significant changes in the predictor constructed, then bagging can improve accuracy." --- Briedman,1996

决策树是Unstable的模型，而KNN是Stable的。

### 0x04. Boosting -- Learn from failures

Boosting方法的基本思想是：

- 给每个样例点分配一个权重
- 迭代T轮，每一轮都把误分类的样本权重提升 -- Focus on the "HARD" ones

#### 1. AdaBoost算法

- 一开始给每个样例点都分配同样的权重 ![1/N ](https://www.zhihu.com/equation?tex=1%2FN%20)  

- for $t = 1,2,...T$ Do:

  - 生成分类器 ![C_t](https://www.zhihu.com/equation?tex=C_t)
  - 计算在此分类器下的错误率 ![\epsilon_t](https://www.zhihu.com/equation?tex=%5Cepsilon_t)所有被误分类的样例权重之和
  - $\alpha_t = \frac{1}{2}ln\frac{1-\epsilon_t}{\epsilon_t}$ (若 ![\epsilon_t > 50\%](https://www.zhihu.com/equation?tex=%5Cepsilon_t%20%3E%2050%5C%25) ,则 ![\alpha_t < 0](https://www.zhihu.com/equation?tex=%5Calpha_t%20%3C%200) ;否则 ![\alpha_t > 0](https://www.zhihu.com/equation?tex=%5Calpha_t%20%3E%200) )
  - 更新样例的权重：
    - 对于正确分类的样例， ![W_{new} = W_{old}*e^{-\alpha_t}](https://www.zhihu.com/equation?tex=W_%7Bnew%7D%20%3D%20W_%7Bold%7D*e%5E%7B-%5Calpha_t%7D) (若 $\epsilon_t>50\%$,则正确样本的权重上升；反之减少)
    - 对分类错误的样例， ![W_{new} = W_{old}*e^{\alpha_t}](https://www.zhihu.com/equation?tex=W_%7Bnew%7D%20%3D%20W_%7Bold%7D*e%5E%7B%5Calpha_t%7D)(若 ![\epsilon_t](https://www.zhihu.com/equation?tex=%5Cepsilon_t) <50%, 则错误样本权重上升)

  - 把所有样例的权重归一化

- 把所有分类器 ![C_t](https://www.zhihu.com/equation?tex=C_t) 的输出结果按照 ![\alpha_t](https://www.zhihu.com/equation?tex=%5Calpha_t) 进行加权平均



#### 2. AdaBoost.M1

- 一开始给每个样例点都分配同样的权重 ![1/N ](https://www.zhihu.com/equation?tex=1%2FN%20)
- for $t = 1,2,...T$ Do:
  - 生成分类器 ![C_t](https://www.zhihu.com/equation?tex=C_t)
  - 计算在此分类器下的错误率 ![\epsilon_t](https://www.zhihu.com/equation?tex=%5Cepsilon_t)=所有被误分类的样例权重之和. ![{\color{blue}{若\epsilon_t>50\%，则终止！}}](https://www.zhihu.com/equation?tex=%7B%5Ccolor%7Bblue%7D%7B%E8%8B%A5%5Cepsilon_t%3E50%5C%25%EF%BC%8C%E5%88%99%E7%BB%88%E6%AD%A2%EF%BC%81%7D%7D)
  - ${\color{blue}{\beta_t = \frac{\epsilon_t}{1-\epsilon_t}}}$ (若 ![\epsilon_t > 50\%](https://www.zhihu.com/equation?tex=%5Cepsilon_t%20%3E%2050%5C%25) ,则 ![\beta_t>1](https://www.zhihu.com/equation?tex=%5Cbeta_t%3E1);否则 ![\beta_t <1](https://www.zhihu.com/equation?tex=%5Cbeta_t%20%3C1))
  - 更新样例的权重：
    - 对于正确分类的样例， ![{\color{blue}{W_{new} = W_{old}*\beta_t}}](https://www.zhihu.com/equation?tex=%7B%5Ccolor%7Bblue%7D%7BW_%7Bnew%7D%20%3D%20W_%7Bold%7D*%5Cbeta_t%7D%7D) (若![\epsilon_t](https://www.zhihu.com/equation?tex=%5Cepsilon_t)> 50%,则正确样本的权重上升；反之减少。总之，增加少数派的权重)
    - 对分类错误的样例， ![{\color{blue}{W_{new} = W_{old}}}](https://www.zhihu.com/equation?tex=%7B%5Ccolor%7Bblue%7D%7BW_%7Bnew%7D%20%3D%20W_%7Bold%7D%7D%7D)(若 ![\epsilon_t](https://www.zhihu.com/equation?tex=%5Cepsilon_t)< 50%,则错误样本权重上升)
- - 把所有样例的权重归一化
- 把所有分类器 ![C_t](https://www.zhihu.com/equation?tex=C_t)C_t 的输出结果按照 ![{\color{blue}{log(1/\beta_t)}}](https://www.zhihu.com/equation?tex=%7B%5Ccolor%7Bblue%7D%7Blog(1%2F%5Cbeta_t)%7D%7D){\color{blue}{log(1/\beta_t)}} 进行加权平均



#### 3. Adaboost运行示例

![img](https://pic3.zhimg.com/v2-98232feadd6675c597937ff9bd658b4a_b.png)

step1：每个样例点都有相同权重

![img](https://pic4.zhimg.com/v2-1206bd42ff08dbcd9a2b419af0584f73_b.png)

step2：第一个模型正确率>50%，所以增加误分类样本的权重

![img](https://pic1.zhimg.com/v2-e07341d787d23254c3fdd6266367c558_b.png)

step3: 提高权重之后，线性分类器发生了变化。现在左侧的三个负例分类错误，所以再提高它们的权重

![img](https://pic3.zhimg.com/v2-0b810ba157a086d1b3688ef1fcd8947a_b.png)

step4：改变权重之后，现在的线性分类器又发生了变化

![img](https://pic3.zhimg.com/v2-ad7c1beb75909053da595033e0dc7e0a_b.png)

step5：最终的分类器是之前若干次分类器按照alpha_t的加权平均，形成了一个**更复杂**的分类器

#### 4. 一些说明

1）如果弱分类器太过复杂，会导致boost之后形成更加复杂的分类器，产生overfitting问题

2）一些模型的包不能使用样例的权重作为参数训练，那么怎么做boosting训练呢？其实，我们完全可以用resampling来代替reweighting. 只要按照权重等比例采样，就可以达到和reweighting类似的效果 (Draw a bootstrap sample from the data with the probability of drawing each example is proportional to it's weight)

------



## 【集成学习常见面试题】

1）集成学习的算法有哪几类，简述其主要思想。

![img](https://pic1.zhimg.com/v2-f0866218e5c385cc8c2d0fe5e3e0d908_b.png)

- Bagging 的全称是 Bootstrap Aggregating，Bagging 采用有放回的采样。通过采样得到多个数据集，并在每个数据集上训练一个基分类器，然后将各分类器的结果结合起来得到最终预测结果（一般分类采取简单投票，回归采取简单平均）。

![img](https://pic4.zhimg.com/v2-c61ad9e98a5b50354169456b5b34e14b_b.png)

- Boosting 也就是所谓的提升方法，多数提升方法是首先改变训练数据的权值，针对不同的训练数据权值调用弱学习算法学习多个弱分类器，然后采用加权多数表决（即加大误差小的弱分类器的权值，减小误差大的弱分类器的权值）的方法将各弱学习器结合在一起。

![img](https://pic2.zhimg.com/v2-a85f3a1d9a19b6417b140d90f8eb5ec9_b.png)



- Stacking是将不同模型的训练结果作为另一个模型的输入，然后得到最终的结果。也就是说，在此类算法中的学习算法可以分为两类，一类是以原数据为输入训练弱学习器的算法，另一类学习算法是以弱学习的输出为输入训练**元学习器**的算法，元学习器学习的其实是各个弱学习器的结合策略。



2）集成学习的优缺点

优点：

- 提升模型准确率
- 更**稳定**、更鲁棒的模型(The aggregate result of multiple models is always less noisy than the individual models. This leads to model stability and robustness.)
- 能够提供非线性(This can be accomplished by using 2 different models and forming an ensemble of the two.)

缺点：

- 可解释性下降
- 计算耗时
- 难以调参



2）Bagging 和 Boosting 两者比较？

① 从并行化上来看：

- Bagging：各个预测模型可以并行生成，因为它们是相互独立的
- Boosting：各个预测函数只能串行生成，因为后一个模型参数需要前一轮模型的结果

②从偏差/方差trade-off来看：

- Bagging：主要减小模型的方差（Variance）。Bagging 就是在几乎不改变模型准确性的前提下尽可能减小模型的方差。因此 Bagging 中的基模型一定要为强模型，否则就会导致整体模型的偏差大，即准确度低。
- Boosting：主要减小模型的偏差（Bias）。Boosting 就是在几乎不改变模型方差的前提下减小模型的偏差。故 Boosting 中的基模型一定要为弱模型，否则就会导致整体模型的方差大（强模型容易过拟合，导致方差过大）。

![img](https://pic3.zhimg.com/v2-43de1f4cee4d350fc741b1cf01f75d3a_b.png)



3）简要介绍 Stacking 算法

Stacking 算法指训练一个**元模型**用于组合各个基模型。具体来说就是将训练好的各个基模型的输出作为元模型的输入来训练一个元模型，这样就能得到一个最终的输出。

具体的过程如下：

1. 划分训练数据集为两个不相交的集合。
2. 在第一个集合上训练多个学习器。
3. 在第二个集合上测试这几个学习器
4. 把第三步得到的预测结果作为输入，把正确的回应作为输出，训练一个高层学习器(元模型)。

![img](https://pic1.zhimg.com/v2-6d72c0afd9a94f2342e2fd3e3dbff6ac_b.png)

按照k折交叉验证的方法（例如k=5），对于model1，每次都在4/5的数据上训练，预测那1/5的数据，这样训练5个model1，最后把它们的输出拼起来，这就是一个完整测试集上的输出了。这样同样训练model2、model3、...model10，得到10个完整测试集输出（10列）。将这10列作为特征，真实标签作为label，来训练元模型（stacking 模型）。

我的理解：Stacking相当于一种订正，把原始模型的输出继续订正，就像后处理(post-processing)一样。



------

​                                                                                             ❤

参考资料：

[1] 清华大学2020春《机器学习概论》课件

[2] 

https://towardsdatascience.com/simple-guide-for-ensemble-learning-methods-d87cc68705a2towardsdatascience.com




  