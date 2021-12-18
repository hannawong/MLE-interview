# GBDT



## 0x01. Boosting方法

​    前面讲过，不同于Bagging, 提升(Boosting)方法通过改变训练样本的**权重**，学习多个分类器，并将这些分类器进行线性组合，提高分类的性能。之前讲过Adaboost算法就是一种典型的boosting方法。

​    历史上，Kearns和Valiant首先提出了**“强可学习**（strongly learnable）”和“**弱可学习**（weakly learnable）”的概念。指出：在概率近似正确（probably approximately correct，PAC）学习的框架中，一个概念（一个类），如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么就称这个概念是强可学习的；一个概念，如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么就称这个概念是弱可学习的。非常有趣的是Schapire后来证明**强可学习与弱可学习是等价的**，也就是说，在PAC学习的框架下，一个概念是强可学习的充分必要条件是这个概念是弱可学习的。

​    这样一来问题便成为：**如果已经发现了“弱学习算法”，那么能否将它提升（boost）为“强学习算法”**。显然，发现弱学习算法通常要比发现强学习算法容易得多。那么如何具体实施提升，便成为开发提升方法时所要解决的问题。关于提升方法的研究很多，有很多算法被提出。最具代表性的是AdaBoost算法。



## 0x02. 提升树

AdaBoost算法只能解决分类问题，因此存在局限性。我们需要寻找一种更普遍的提升方法，来解决不同场景下的问题：

- **分类问题：**使用交叉熵损失函数

![img](https://pic4.zhimg.com/v2-693c3f7b33f5dfd6ca87b46e92e4186f_b.png)

p:label, q: prediction; 当p和q分布差异较大时，交叉熵损失也大；当p和q分布完全一致，交叉熵损失为0

- **回归问题：**使用平方损失函数
- **一般决策问题：**任意损失函数

提升树就是这样一种可以解决不同场景下问题的boosting方法。

#### 2.1 分类问题

对于分类问题，提升树算法实际上是AdaBoost算法的特殊情况（即基分类器为分类树），这里不再累述。

#### 2.2 一般决策问题（包括回归问题）

   对于回归问题和一般决策问题，在这里放在一起来讲（因为回归问题就是一般决策问题的一个特例）。这两种问题都需要使用前向分步算法，具体的操作为：

   首先，训练第一棵树。然后在第一棵树的基础上增加第二棵树，我们希望增加第二棵树之后可以降低loss。

![img](https://pic3.zhimg.com/v2-0678bed836a12df1f64b4e53b058dfba_b.png)

​                                                  

​                                                  (我们的目标：增加第二棵树之后，loss降低)

​    我们只需要在第一棵树的基础上，找到梯度下降最快的方向，然后移动 $\eta$ 即可。

![img](https://pic3.zhimg.com/v2-d9cda0a854d015354034c8debc13c27a_b.png)

​                        (左图：第一颗树的输出在loss曲线上的位置；右图：向梯度下降最快的方向移动)

​    在数学上，只需要计算$-\frac{\partial loss}{\partial 上一棵树的输出}$ 作为梯度负方向，然后* $\eta$ 作为参数更新的公式。也就是，用第m棵树去拟合这个**梯度负方向**。

第二棵树去拟合 $-\frac{\partial L}{\partial F(1)}$, ... 第m棵树去拟合$-\frac{\partial L}{\partial F(m-1)}$:

![img](https://pic1.zhimg.com/v2-6e1531a97c8dc084f0bf37f8b9417068_b.png)

​                                                                        (第二棵树的计算方法)

![img](https://pic3.zhimg.com/v2-d098afdbd7352eb6ea7a33b3795e268e_b.jpeg)

​                                                                         (第m棵树的计算方法)

> This is nothing but the gradient of the loss function with respect to the output of the previous model. That's why this method is called Gradient Boosting.

​    相比于随机森林，GBDT有着**更大的模型capacity**，可以拟合非常复杂的函数。但是，和那些有着高模型capacity的其他模型一样，GBDT也会很快的过拟合，所以要小心这一点！

​    特殊地，对于回归问题（**平方误差损失**），第m次迭代时损失函数为$\frac{1}{2}(y-F(m-1))^2$ ,计算负梯度 $-\frac{\partial loss}{\partial F(m-1)} = -\frac{\partial \frac{1}{2}(y-F(m-1))^2}{\partial F(m-1)} = y-F(m-1)​$ ,就是第m-1棵树的**残差**。所以，对于平方误差损失的回归问题，就是用树去不断拟合上一棵树的残差；而对于一般的问题，就是用树**不断地拟合上一棵树的负梯度**。