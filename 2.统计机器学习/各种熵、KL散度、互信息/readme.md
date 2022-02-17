##### 1. 香农信息量

概率越小的事情发生了，那么这件事的信息量越大。（比如”太阳从东边升起“这句话信息量为0，因为这个事件概率为1；但是”今天有陨石掉落“这句话的信息量很大。）

香农信息量定义为:         ![-log_2p(x)](https://www.zhihu.com/equation?tex=-log_2p(x))  

##### 2.熵
对于整个系统而言，我们更加关心的是表达系统整体所需要的信息量。熵表示整个系统的混乱程度：

​                                                      ![H(p)=-\sum _{i=1}^n p(X=x_i)log_2 p(X = x_i)](https://www.zhihu.com/equation?tex=H(p)%3D-%5Csum%20_%7Bi%3D1%7D%5En%20p(X%3Dx_i)log_2%20p(X%20%3D%20x_i))  

如果n个事件都是等概率发生，那么混乱程度最大，熵也最大。

##### 3.联合熵
两个随机变量X，Y的联合分布，可以形成联合熵，用H(X,Y)表示。

​                                                 ![H(X,Y) = \sum_{i,j} p(X=x_i, Y=y_j)log_2p(X=x_i, Y=y_j)](https://www.zhihu.com/equation?tex=H(X%2CY)%20%3D%20%5Csum_%7Bi%2Cj%7D%20p(X%3Dx_i%2C%20Y%3Dy_j)log_2p(X%3Dx_i%2C%20Y%3Dy_j))  

##### 4. 条件熵
在随机变量Y发生的前提下，随机变量X的信息熵。用来衡量在**已知随机变量Y的条件下**随机变量X的不确定性。

​                                                  ![H(X,Y) = \sum_{i,j} p(X=x_i, Y=y_j)log_2p(X=x_i| Y=y_j)](https://www.zhihu.com/equation?tex=H(X%2CY)%20%3D%20%5Csum_%7Bi%2Cj%7D%20p(X%3Dx_i%2C%20Y%3Dy_j)log_2p(X%3Dx_i%7C%20Y%3Dy_j))  

##### 5.相对熵(KL散度)

KL 散度是一种衡量两个分布之间的匹配程度的方法。

KL散度在形式上定义如下：

​                                                     ![D_{KL}(p||q) = \sum _{i=1}^n p(X=x_i)log \frac{p(X=x_i)}{q(X=x_i)}](https://www.zhihu.com/equation?tex=D_%7BKL%7D(p%7C%7Cq)%20%3D%20%5Csum%20_%7Bi%3D1%7D%5En%20p(X%3Dx_i)log%20%5Cfrac%7Bp(X%3Dx_i)%7D%7Bq(X%3Dx_i)%7D)  


其中 q(x) 是近似分布，p(x) 是真实分布。直观地说，这衡量的是给定**任意分布偏离真实分布的程度**。如果两个分布完全匹配，那么 ![D_{KL}(p||q)=0](https://www.zhihu.com/equation?tex=D_%7BKL%7D(p%7C%7Cq)%3D0)  。KL 散度越小，真实分布与近似分布之间的匹配就越好。

**公式的直观解释:**
让我们看看 KL 散度各个部分的含义。首先看看 ![log \frac{p(X=x_i)}{q(X=x_i)}](https://www.zhihu.com/equation?tex=log%20%5Cfrac%7Bp(X%3Dx_i)%7D%7Bq(X%3Dx_i)%7D)  项，如果 ![q(X=x_i) == p(X=x_i)](https://www.zhihu.com/equation?tex=q(X%3Dx_i)%20%3D%3D%20p(X%3Dx_i))  则该项的值为 0。然后，为了使这个值为期望值，要用 ![p(X=x_i)](https://www.zhihu.com/equation?tex=p(X%3Dx_i))  来给这个对数项加权。也就是说 ![p(X=x_i)](https://www.zhihu.com/equation?tex=p(X%3Dx_i))  有更高概率的匹配区域比低  ![p(X=x_i)](https://www.zhihu.com/equation?tex=p(X%3Dx_i))  概率的匹配区域更加重要。

**KL散度的性质：**

- 不对称性

尽管KL散度从直观上是个度量或距离函数，但它并不是一个真正的度量或者距离，因为它不具有对称性，即 ![D(P||Q) \neq D(Q||P)](https://www.zhihu.com/equation?tex=D(P%7C%7CQ)%20%5Cneq%20D(Q%7C%7CP))  。这是因为KL散度是针对**近似分布偏离真实分布的程度**来说的。

- 非负性

相对熵的值是非负值，即 ![D(P||Q)>0](https://www.zhihu.com/equation?tex=D(P%7C%7CQ)%3E0)  。

**【JS散度】**

JS 散度度量了两个概率分布的相似度，基于 KL 散度的变体，**解决了 KL 散度非对称的问题**。一般地，JS 散度是对称的，其取值是 0 到 1 之间。

定义如下：

![img](https://cdn.hyper.ai/wp-content/uploads/2019/01/vcw7xahr.png)



##### 6. 交叉熵
和KL散度的关系:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217191926481.png)

等式的前一部分恰巧就是p的熵，等式的后一部分，就是交叉熵：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217191952131.png)

其中， ![p(x_i)](https://www.zhihu.com/equation?tex=p(x_i))  是真实分布,  ![q(x_i)](https://www.zhihu.com/equation?tex=q(x_i))  是预测分布。

在机器学习中，我们需要评估label和predicts之间的分布差异，所以使用KL散度最为合适，即 ![D_{KL}(y||\hat y)](https://www.zhihu.com/equation?tex=D_%7BKL%7D(y%7C%7C%5Chat%20y))  。由于KL散度中的前一部分−H(y)不变，故在优化过程中，只需要关注交叉熵就可以了。所以一般在机器学习中直接用交叉熵做loss，评估模型。

例如，对于单分类问题：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217192429103.png)


​                                         ![loss=−(0×log(0.3)+1×log(0.6)+0×log(0.1)=-log(0.6)](https://www.zhihu.com/equation?tex=loss%3D%E2%88%92(0%C3%97log(0.3)%2B1%C3%97log(0.6)%2B0%C3%97log(0.1)%3D-log(0.6))  

对于多分类问题：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217192512891.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210217192533341.png)

这里不采用softmax(因为概率总和不再是1)，而是采用sigmoid把每个概率值放缩到(0,1)即可。

单个样本的loss即为 $loss=loss_猫+loss_蛙+loss_鼠​$




**Q/A: 为什么要用交叉熵损失而不是MSE？**
https://blog.csdn.net/yhily2008/article/details/80261953
其实，简言之就是交叉熵是KL散度的一项，就是用来度量两个分布的差异的。而普通的MSE不能做到这点。

##### 7. 互信息
Mutual Information表示两个变量$X$与$Y$是否有关系，以及关系的强弱，因此可以用于特征的筛选。形象的解释是：假如$X$是一个随机事件，$Y$也是一个随机事件，那么$X$和$Y$相互依赖的程度应该是：

不知道X时，Y发生的不确定性（熵）- 已知X时，Y发生的不确定性(熵)。即：

​                                       ![I(X;Y)=H(Y)-H(Y|X) = H(X)-H(X|Y)](https://www.zhihu.com/equation?tex=I(X%3BY)%3DH(Y)-H(Y%7CX)%20%3D%20H(X)-H(X%7CY))  


