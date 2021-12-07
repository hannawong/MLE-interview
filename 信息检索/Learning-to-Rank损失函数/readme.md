#### LTR：Pointwise, Pairwise, Listwise

对于一个query q和document d，我们需要用神经信息检索的方法来计算q和d的相似程度 ![f(q,d)](https://www.zhihu.com/equation?tex=f(q%2Cd)).

![img](https://pic4.zhimg.com/v2-7ed50e1422f2e97b5dd1c3a6e4554cdb_b.png)

但是对于基于排序的评价指标(**MAP,NDCG,MRR**)通常是**不可微**的，不能直接优化。所以需要一些可微的损失函数来优化Learning-to-Rank神经网络。

这对于LTR一般说来有三类方法：**Pointwise, Pairwise和Listwise**。

**1.Pointwise**

这个非常简单，和从前见到的分类/回归问题别无二致。预测模型给出的相似度为 ![f(q,d)](https://www.zhihu.com/equation?tex=f(q%2Cd)) ,"真实"的相似度可以由用户是否点击这样的implicit feedback来打标签。损失函数可以是类似MSE这样的回归损失函数 ![loss = ||y-f(q,d)||^2](https://www.zhihu.com/equation?tex=loss%20%3D%20%7C%7Cy-f(q%2Cd)%7C%7C%5E2) ，也可以是分类问题的交叉熵损失函数。

**2.Pairwise: Predict pairwise preference between documents for a query.**

![Loss = max\{0,m +f(q,d_-)-f(q,d_+)\}](https://www.zhihu.com/equation?tex=Loss%20%3D%20max%5C%7B0%2Cm%20%2Bf(q%2Cd_-)-f(q%2Cd_%2B)%5C%7D)如果负样本计算出来的相关性 + m > 正样本计算出的相关性，那么loss > 0，意味着模型把非正确的回答排在正确答案的上面；如果![L](https://www.zhihu.com/equation?tex=L)等于0，模型把正确的回答排在非正确的回答之上。总之该**hinge损失函数**的目的就是促使正确答案的得分比错误答案的得分高至少![m](https://www.zhihu.com/equation?tex=m)。和pointwise类似，在预测阶段按照得分排序。

在训练阶段，需要有query和正负样本对，所以Pairwise方法涉及到**【负样本采样】**，这个有很多文章研究过，之后会详细说明。

**3.Listwise方法**

复杂度过高，现实中几乎不会使用。