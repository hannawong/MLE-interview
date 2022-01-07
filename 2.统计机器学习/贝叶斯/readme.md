#  贝叶斯推断



### 1. 概述

贝叶斯定理 (Bayes’ theorem) 表达了一个事件发生的概率，而确定这一概率的方法是基于与该事件相关的条件**先验知识 (prior knowledge)**。而利用相应先验知识进行概率推断的过程为贝叶斯推断 (Bayesian inference)。



### 2. 贝叶斯定理

D是{“咳嗽”,“味觉消失”,“发烧”}，h是“得了流感”。

P(h): 得了流感的先验概率

P(D): 出现症状{“咳嗽”,“味觉消失”,“发烧”}的**先验概率**

P(h|D): 得了流感的**后验概率** ，即在出现症状这个事件发生之后，对“得了流感”事件概率的重新评估。

P(D|h): **似然度** likelihood，实际经常取log，因为概率相乘会非常小。



> **先验概率**：是指现有数据根据以往的经验和分析得到的概率
>
> **后验概率**：事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小



#### 2.1 极大后验假设(Maximum A posteriori)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220143837923.png)



可以看到第二个等式到第三个等式时，忽略了分母上的先验概率P(D):
![img](https://img-blog.csdnimg.cn/20201220143852647.png)



### 2.3 极大似然假设(Maximum Likelihood)
​                                                                ![h_{ML} = argmax_{h \in H} P(D|h)](https://www.zhihu.com/equation?tex=h_%7BML%7D%20%3D%20argmax_%7Bh%20%5Cin%20H%7D%20P(D%7Ch))  

其中，h为标签，D为样本特征。
可见，在极大似然假设是忽略了先验概率P(h). 当P(h)未知或者等概率的时候可以这样做。



##### 2.3.1 maximum likelihood 和 Least square error 等价
设有n个样本 ![\{x_1, x_2,... x_n\}](https://www.zhihu.com/equation?tex=%5C%7Bx_1%2C%20x_2%2C...%20x_n%5C%7D)  , 它们的真实标签值 ![\{d_1,d_2,..d_n\}](https://www.zhihu.com/equation?tex=%5C%7Bd_1%2Cd_2%2C..d_n%5C%7D)  . 使用函数 ![h(x)](https://www.zhihu.com/equation?tex=h(x))  来拟合，它是一个无噪音的目标函数。假设噪音 ![e_i](https://www.zhihu.com/equation?tex=e_i)  是独立的随机变量，符合正态分布 ![N(0,\sigma^2)](https://www.zhihu.com/equation?tex=N(0%2C%5Csigma%5E2))  , 那么有：  ![d_i=h(x_i)+e_i ](https://www.zhihu.com/equation?tex=d_i%3Dh(x_i)%2Be_i%20)  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210216195221951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

 ![d_i](https://www.zhihu.com/equation?tex=d_i)  也服从正态分布  ![N(h(x_i),\sigma^2)](https://www.zhihu.com/equation?tex=N(h(x_i)%2C%5Csigma%5E2))  .那么，第 i 个样本的标签为 ![d_i](https://www.zhihu.com/equation?tex=d_i)  的概率为： 
![img](https://pic3.zhimg.com/80/v2-d9c12f9f641c2d2de76b5ab7d6543044_1440w.png)

所有样本标签为 ![\{d_1,d_2,..d_n\}](https://www.zhihu.com/equation?tex=%5C%7Bd_1%2Cd_2%2C..d_n%5C%7D)  的概率为：

![1641514386533](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1641514386533.png)

两边取对数：

![1641514427991](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1641514427991.png)

最大似然法就是要取一个合适的函数 h(x) 来最大化这个概率，即：![1638999671217](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1638999671217.png)



所以，极大似然和最小均方误差等价的充要条件是：误差满足正态分布。



### 3. 朴素贝叶斯分类器

 ![h_{ML} = argmax_{h \in H} P(D|h)](https://www.zhihu.com/equation?tex=h_%7BML%7D%20%3D%20argmax_%7Bh%20%5Cin%20H%7D%20P(D%7Ch))  

朴素贝叶斯引入的假设是独立性假设, 假设自变量之间是独立的：

​                                         ![P(D∣h_i)=P(d_1,d_2,...d_n∣h_i)=\prod_j P(d_j∣h_i)](https://www.zhihu.com/equation?tex=P(D%E2%88%A3h_i)%3DP(d_1%2Cd_2%2C...d_n%E2%88%A3h_i)%3D%5Cprod_j%20P(d_j%E2%88%A3h_i))  

朴素贝叶斯分类器：

![1638999998682](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1638999998682.png)

取log是为了防止连乘导致的数据太小，产生误差。取log之后变成了加法，解决了这个问题。



## 4. 面试题

#### 0x01: 什么是贝叶斯决策论？

首先得先了解以下几个概念：**先验概率、条件概率、后验概率、误判损失、条件风险、贝叶斯判别准则**。下面我们一个个来进行讨论。

**先验概率：** 所谓先验概率，就是根据以往的经验或者现有数据的分析所得到的概率。如，得流感的概率为3%。

**条件概率：**P(B|A)，即B在A发生的条件下发生的概率。比如，得了流感的前提下，出现嗓子疼症状的概率是90%。条件概率是"**有因求果**"。

**后验概率：** 数学表达式为p(A|B), 即A在B发生的条件下发生的概率。比如，在出现"嗓子疼"这个症状的条件下，得流感的概率为50%。这就是后验概率，后验概率是**有果求因**（知道结果推出原因）

**误判损失:** L(j|i)，表示把一个标记为​ i 类的样本误分类为 j 类所造成的损失。

**条件风险:** ∑L(i|j)P(j|x)。其实就是所有误判损失的加权和，而这个权就是样本判为 j 类的概率.

**贝叶斯判别准则：** 贝叶斯判别准则是找到一个使**条件风险达到最小**的判别方法。即，将样本判为哪一类，所得到的条件风险（或者说平均判别损失）最小，那就将样本归为那个造成平均判别损失最小的类。



#### 0x02. “朴素”是朴素贝叶斯在进行预测时候的缺点，那么有这么一个明显的假设缺点在，为什么朴素贝叶斯的预测仍然可以取得较好的效果？

答：**"朴素"指的是独立性架设**。对于分类任务来说，只要各个条件概率之间的**排序**正确，那么就可以通过比较概率大小来进行分类，不需要知道精确的概率值(朴素贝叶斯分类的核心思想是找出后验概率最大的那个类，而不是求出其精确的概率)。
如果属性之间的相互依赖对所有类别的影响相同，或者相互依赖关系可以互相抵消，那么属性条件独立性的假设在降低计算开销的同时不会对分类结果产生不良影响。



#### 0x03. 什么是拉普拉斯平滑法?

拉普拉斯平滑法是朴素贝叶斯中处理**零概率问题**的一种修正方式。在进行分类的时候，可能会出现某个属性在训练集中没有与某个类同时出现过的情况，如果直接基于朴素贝叶斯分类器的表达式进行计算的话就会出现零概率现象, 导致整个概率全为0. 为了避免其他属性所携带的信息被训练集中未出现过的属性值“抹去”，所以才使用拉普拉斯估计器进行修正。

#### 0x04. 朴素贝叶斯是高方差还是低方差模型？
朴素贝叶斯是低方差模型(误差 = 偏差 + 方差).对于复杂模型来说，由于复杂模型充分拟合了部分数据，使得它们的偏差变小，但由于对部分数据过分拟合，这就导致预测的方差会变大。因为朴素贝叶斯假设了各个属性之间是相互独立，算是一个简单的模型。对于简单的模型来说，偏差更大、方差较小。(偏差是模型输出值与真实值的误差，也就是模型的精准度，方差是预测值与模型输出期望的的误差，即模型的稳定性，也就是数据的集中性的一个指标)。

#### 0x05. 单词纠错

经常在网上搜索东西的朋友知道，当你不小心输入一个不存在的单词时，搜索引擎会提示你是不是要输入某一个正确的单词，比如当你在Google中输入“computet”时，系统会猜测你的意图：是不是要搜索“computer”，如下图所示：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217113718834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



Google的拼写检查基于贝叶斯方法。请说说的你的理解，具体Google是怎么利用贝叶斯方法，实现"拼写检查"的功能。

答："拼写检查"要做的事情就是：在输入一个错误词语w的情况下，试图推断出正确词语c。换言之：已知w，然后在若干个备选方案中，找出可能性最大的那个c，也就是求P(c|w)的最大值。
而根据贝叶斯定理，有：         

​                                                                      ![P(c|w) = \frac{P(w|c)P(c)}{P(w)} ](https://www.zhihu.com/equation?tex=P(c%7Cw)%20%3D%20%5Cfrac%7BP(w%7Cc)P(c)%7D%7BP(w)%7D%20)                                                                                                                                                        

由于对于所有备选的 c​ 来说，对应的都是同一个​ w​ ，所以它们的​ P(w)​ 是相同的，因此我们只要最大化P(w|c)P(c)即可。其中：

P(c)​ 表示某个正确的词的出现"概率"，它可以用"频率"代替。如果我们有一个足够大的文本库，那么这个文本库中每个单词的出现频率，就相当于它的发生概率。某个词的出现频率越高, P(c)就越大。比如在你输入一个错误的词“computet”时，系统更倾向于去猜测你可能想输入的词是“computer”，而不是“computeo”，因为“computer”更常见。
P(w∣c)表示在试图拼写c的情况下，出现拼写错误w的概率。为了简化问题，假定两个单词在字形上越接近，就有越可能拼错，P(w|c)就越大。举例来说，相差一个字母的拼法，就比相差两个字母的拼法，发生概率更高。值得一提的是，一般把这种问题称为“**编辑距离**”。







