### 1. 线性回归

基本假设：**误差是均值为0的正态分布**

![img](https://pic1.zhimg.com/80/v2-6d056385519566a4722c02f86cbebabb_1440w.png)

目标就是拟合 ![W,b](https://www.zhihu.com/equation?tex=W%2Cb)  ；在损失函数为MSE时，此问题就是最小二乘问题，有close-form solution:

​                                                          ![W = (X^TX)^{-1}X^Ty](https://www.zhihu.com/equation?tex=W%20%3D%20(X%5ETX)%5E%7B-1%7DX%5ETy)  

### 2. Bias-Variance Tradeoff

![img](https://pica.zhimg.com/80/v2-35d08dc52ec8130ecfd53ff90f8a07fa_1440w.png)

复杂模型：high variance, low bias;

简单模型：low variance, high bias

我们常常需要**【正则化方法】**来解决过拟合问题。这里提一下L1和L2正则化方法。

### 2.1 L2正则化 --- ridge regression

这样做的好处就是使参数的weight不要太大。例如，使用多项式模型，如果使用 10 阶多项式，模型可能过于复杂，容易发生过拟合。所以，为了防止过拟合，我们可以将其高阶部分的权重 w 限制为 0，这样，就相当于从高阶的形式转换为低阶。

为了达到这一目的，最直观的方法就是限制 w 的个数，但是这类条件属于 NP-hard 问题，求解非常困难。所以，一般的做法是寻找更宽松的限定条件，即参数平方和<c的限制：

![img](https://pica.zhimg.com/80/v2-516c9054cd869303df5e14bf49dbd236_1440w.png)

损失函数增加L2正则项，即参数的二范数：

![img](https://pic2.zhimg.com/80/v2-e52976e32aa66a62184e1031386d297a_1440w.png)

其实，ridge regression也是有close-form solution的：

​                                                           ![W = (X^TX+\alpha I)^{-1}X^Ty](https://www.zhihu.com/equation?tex=W%20%3D%20(X%5ETX%2B%5Calpha%20I)%5E%7B-1%7DX%5ETy)  

### 2.2 L1 正则化 -- Lasso Regression

增加参数绝对值和<c的限制：

![img](https://pic2.zhimg.com/80/v2-dac61d6772c055a5cb84f6bc3f32796f_1440w.png)

损失函数增加L1损失项，即参数的1-范数，用此损失函数没有close-form solution：

![img](https://pic1.zhimg.com/80/v2-9a34f82c1105ab8e94c69cd5a5094218_1440w.png)

### 2.3 Lasso和Ridge Regression的对比

简言之，L1正则化可以造成特征的**稀疏性**，因此可以做特征选择；L2正则化不能。

![img](https://pic2.zhimg.com/80/v2-88998c93500e64b073ebad3004ede58f_1440w.png)

 ![\hat{\beta}](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cbeta%7D)  是能够达到最小化MSE的位置，其周围的等高线为不同的MSE值；蓝色菱形和圆形分别为L1和L2正则化的约束区间，那么只有等高线与它们的交点才是最优的参数位置。所以，L1正则可以让某些参数为0，起到了特征选择的效果。

### 2.4 ElasticNet--结合Lasso & Ridge

限制：

![img](https://pic1.zhimg.com/80/v2-84ee4b71a0b4a4f4e88560c862e05017_1440w.png)

![img](https://pic3.zhimg.com/80/v2-f4b5a7290d8dd7e554e8bdfee8c4e59b_1440w.png)

损失函数：

![img](https://pic1.zhimg.com/80/v2-d9a9f887cf7bf3b48792da598418b297_1440w.png)



#### 更深入理解L1/L2正则

##### 拉普拉斯先验和正态先验

**先验**的意思是对一种未知的东西的假设，比如说我们看到一个正方体的骰子，那么我们会假设他的各个面朝上的概率都是1/61/6，这个就是先验。但事实上骰子的材质可能是密度不均的，所以还要从数据集中学习到更接近现实情况的概率。同样，在机器学习中，我们会根据一些已知的知识**对参数的分布进行一定的假设**，这个就是先验。有先验的好处就是可以在较小的数据集中有良好的**泛化**性能，当然这是在**先验分布是接近真实分布**的情况下得到的了。从信息论的角度看，向系统加入了正确先验这个信息，肯定会提高系统的性能。我们假设参数θ是如下的Laplace分布的，这就是Laplace先验：

![img](https://pic2.zhimg.com/80/v2-e7cd836b9360607ba84a4aedb3eba76b_1440w.png)

其中，$\lambda$是个超参数。laplace分布的图长这样：

![img](https://images2017.cnblogs.com/blog/1191997/201710/1191997-20171018191512974-1322748219.jpg)

带有先验知识的”极大似然估计“就叫做”极大后验估计“了（具体参见贝叶斯一章）。因此，我们要最大化的后验估计还需要乘上一个$P(\theta)$:

![img](https://pica.zhimg.com/80/v2-04571712db49c1b2ed8c065361b25cc1_1440w.png)

带入拉普拉斯先验公式得到：

![img](https://pica.zhimg.com/80/v2-729652ad7555c0c10774d6285b26962c_1440w.png)

这不就是L1正则化的公式吗！所以说，**L1正则化的参数$\theta$先验分布是拉普拉斯分布。**

类似的，**L2正则化是假设参数$\theta$符合高斯分布**：

![img](https://pic2.zhimg.com/80/v2-3a51da8c67f03c6ece4e92414f758bca_1440w.png)

带入式中可以得到L2正则化公式：

![img](https://pic2.zhimg.com/80/v2-e737fdf3b93ca3ed3068577a38b7c8f3_1440w.png)