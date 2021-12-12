# 优化器



## 先修知识：牛顿法

#### 1.牛顿法

##### 1.1 应用一：求方程的根

![img](https://img-blog.csdnimg.cn/20210125183922803.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



##### 1.2 应用二：最优化

![img](https://img-blog.csdnimg.cn/20210426221329818.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



推广到多元的情况，一阶导变为梯度、二阶导变为海森矩阵：

![img](https://img-blog.csdnimg.cn/20210125145603537.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



# 优化器





![img](https://easyai.tech/wp-content/uploads/2019/01/tiduxiajiang-1.png)



## 0x01. 梯度下降法(Gradient Descent)
梯度下降法是最基本的一类优化器，目前主要分为三种梯度下降法：标准梯度下降法(GD, Gradient Descent)，随机梯度下降法(SGD, Stochastic Gradient Descent)及批量梯度下降法(BGD, Batch Gradient Descent)。

##### 1. 标准梯度下降法(GD)
假设要学习训练的模型参数为$\theta$，loss为$J(\theta)$，则loss关于模型参数的偏导数，即梯度为$g_t = \frac{\partial J(\theta)}{\partial \theta}|_{\theta = \theta_{t-1}}$，学习率为$η$，则使用梯度下降法更新参数为：$\Delta \theta = -\eta g_t$

若参数是多元($\theta_1,\theta_2...\theta_n$)的，则梯度为：

![img](https://img-blog.csdnimg.cn/20210217221519422.png)

​      

基本策略可以理解为”在有限视距内寻找最快路径下山“，因此每走一步，参考当前位置最陡的方向(即梯度)进而迈出下一步。可以形象的表示为：

![img](https://img-blog.csdn.net/2018042514342430)




上图中，红色部分代表损失函数 $J(\theta )$ 比较大的地方，蓝色部分是损失函数小的地方。我们需要让$J(\theta )$的值尽量的低，也就是达到深蓝色的部分。$w_1$，$w_2$表示W向量的两个维度。

然而，标准梯度下降法每走一步都要在**整个训练集上**计算调整下一步的方向，下山的速度慢。在应用于大型数据集中，每次迭代都要遍历所有的样本，会使得训练过程及其缓慢。所以，下面介绍随机梯度下降法和小批量梯度下降法。

##### 1.2 随机梯度下降法(SGD)

每次只取**一个样本**计算梯度，并更新权重。这里虽然引入了随机性和噪声，但期望仍然等于正确的梯度下降。

- 优点：虽然SGD需要走很多步，但是计算梯度快。
- 缺点：SGD在随机选择梯度的同时会引入噪声，使得权值更新的方向不一定正确。



##### 1.3 小批量梯度下降(BGD)

每次批量输入BATCH_SIZE个样本，模型参数的调整更新与全部BATCH_SIZE个输入样本的loss函数之和有关。
基本策略可以理解为，在下山之前掌握了附近的地势情况，选择总体平均梯度最小的方向下山。批量梯度下降法比标准梯度下降法训练时间短，且每次下降的方向都很正确。



所有梯度下降方法的缺点都是容易陷入局部最优解：由于是在有限视距内寻找下山的反向，当陷入平坦的洼地，会误以为到达了山地的最低点，从而不会继续往下走。所谓的局部最优解就是**鞍点**。落入鞍点，梯度为0，使得模型参数不再继续更新。



##### Q/A

**Q: 梯度下降法找到的一定是下降最快的方向么？**
A：梯度下降法并不是下降最快的方向，它只是目标函数在当前的点的切平面（当然高维问题不能叫平面）上下降最快的方向。在实际使用中，牛顿方向（考虑海森矩阵）才一般被认为是下降最快的方向。牛顿法是二阶收敛，梯度下降是一阶收敛，前者牛顿法收敛速度更快。

但是为什么在一般问题里梯度下降比牛顿类算法更常用呢？因为对于规模比较大的问题，海塞矩阵计算是非常耗时的；同时对于很多对精度需求不那么高的问题，梯度下降的收敛速度已经足够了。
非线性规划当前的一个难点在于处理非凸问题的全局解，而搜索全局解这个问题一般的梯度下降也无能为力。

 

## 0x02. 动量优化法

动量优化方法是在梯度下降法的基础上进行的改变，具有**加速梯度下降**的作用。一般有标准动量优化方法Momentum、NAG（Nesterov accelerated gradient）动量优化方法。

#### 1. momentum

Momentum的“梯度”不仅包含了这一步实际算出来的梯度，还包括了上一次的梯度“惯性”。其实，动量项$m_t$可以看作$E[g_t] $的移动平均。

![img](https://img-blog.csdnimg.cn/2021012711360045.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



- **下降初期**时，使用上一次参数更新，下降方向一致
- 下降中后期时，在局部最小值来回震荡的时候，![gradient \rightarrow 0](https://private.codecogs.com/gif.latex?gradient%20%5Crightarrow%200)，但是由于具有上一次的动量![m_{t-1}](https://private.codecogs.com/gif.latex?m_%7Bt-1%7D),所以能够跳出陷阱
- 在梯度![g_t](https://private.codecogs.com/gif.latex?g_t)改变方向（震荡）的时候，由于具有上一次的动量，所以会“往回掰”一点，抑制震荡。

总而言之，momentum项能够在原先方向加速SGD，抑制振荡，从而加快收敛。

由于当前梯度的改变会受到**上一次梯度**改变的影响，类似于小球向下滚动的时候带上了惯性。这样可以加快小球向下滚动的速度。

![preview](https://img-blog.csdnimg.cn/img_convert/7fb767587ea8f71fd478e8f8f7ac01be.png)



#### 2. NAG

- 牛顿加速梯度（NAG, Nesterov accelerated gradient）算法，是Momentum动量算法的变种

nesterov项在梯度更新时做一个校正，避免前进太快，同时提高灵敏度。 

![img](https://img-blog.csdnimg.cn/20210127113652693.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



所以，加上nesterov项后，梯度在大的跳跃后，进行计算对当前梯度进行校正。如下图：

![img](https://img-blog.csdnimg.cn/20210127113707743.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

**momentum**首先计算一个梯度(短的蓝色向量)，然后在原先梯度的方向(惯性)进行一个大的跳跃(长的蓝色向量)

**nesterov**项首先在原先梯度的方向进行一个大的跳跃(棕色向量)，计算梯度然后进行**校正**(绿色向量)



## 3.自适应学习率优化算法
自适应学习率优化算法针对于机器学习模型的学习率，传统的优化算法要么将学习率设置为常数要么根据训练次数调节学习率。极大忽视了学习率其他变化的可能性。然而，学习率对模型的性能有着显著的影响，因此需要采取一些策略来想办法更新学习率，从而提高训练速度。
目前的自适应学习率优化算法主要有：AdaGrad算法，RMSProp算法，Adam算法以及AdaDelta算法。

##### 3.1 AdaGrad算法

Adagrad其实是对学习率进行了一个约束。即：
![img](https://img-blog.csdnimg.cn/2021012711373897.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



- 前期$\sum g_r^2$较小的时候，learning rate较大，能够放大梯度
- 后期$\sum g_r^2$较大的时候，learning rate较小，能够约束梯度

缺点：

- 由公式可以看出，仍依赖于人工设置一个全局学习率$\eta$
- 一开始分母太小，所以learning rate太大，对梯度的调节太大
- 而中后期，分母上梯度平方的累加将会越来越大，使学习率趋近于0，使得训练提前结束



##### **3.2 RMSProp算法**

- RMSProp算法修改了AdaGrad的梯度**积累**为指数加权的移动**平均，**避免了学习率越来越低的的问题。
- RMSProp算法在经验上已经被证明是一种有效且实用的深度神经网络优化算法。目前它是深度学习从业者经常采用的优化方法之一。   

![img](https://img-blog.csdnimg.cn/20210127113756534.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

##### **3.3 Adam**

Adam(Adaptive Moment Estimation)本质上是**带momentum的RMSprop**，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围。

![img](https://img-blog.csdnimg.cn/2021012711385648.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



为什么说Adam是"带momentum的RMSprop"呢? 我们把参数更新的公式拆解成这样就容易看清了：

​                                                                    $$\Delta \theta_t = -\frac{\eta}{\sqrt{\hat{n_t}+\epsilon}} \cdot \hat{m_t}$$

其中，左边这一项就是自适应调整学习率的项，分母中的$\hat{n_t}$ 就对应RMSprop中的$E[g_t^2]$. 右边是所谓"动量项"，就是$E[g_t]$的移动平均。 

Adam通常被认为对超参数的选择相当鲁棒，尽管学习率有时需要从建议的默认修改。

 可以看出，直接对梯度的矩估计对内存没有额外的要求，而且可以根据梯度进行动态调整。



##### 3.4 AdamW

BERT中的优化器用的就是AdamW. 

AdamW是在**Adam+L2正则化**的基础上进行改进的算法。
使用Adam优化带L2正则的损失并不有效。如果引入L2正则项，在计算梯度的时候会加上对正则项求梯度的结果。那么如果本身比较大的一些权重对应的梯度也会比较大，由于Adam计算步骤中减去项会有除以梯度平方的累积，使得减去项偏小。按常理说，越大的权重应该惩罚越大，但是在Adam并不是这样。而权重衰减对所有的权重都是采用相同的系数进行更新，越大的权重显然惩罚越大。在常见的深度学习库中只提供了L2正则，并没有提供权重衰减的实现。



![img](https:////upload-images.jianshu.io/upload_images/19036657-526f2e6d75337b2b.png?imageMogr2/auto-orient/strip|imageView2/2/w/689/format/webp)



---

鞍点：

![这里写图片描述](https://img-blog.csdn.net/20180426113728916)



- 三个自适应学习率优化器没有进入鞍点，其中，AdaDelta下降速度最快，Adagrad和RMSprop则齐头并进。
- 两个动量优化器Momentum和NAG以及SGD都顺势进入了鞍点。但两个动量优化器在鞍点抖动了一会，就**逃离了鞍点**并迅速地下降。
- 很遗憾，SGD进入了鞍点，却始终停留在了鞍点，没有再继续下降。
