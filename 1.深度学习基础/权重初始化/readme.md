# 网络权重初始化

### 0x01. 权重初始化为何如此重要？

虽然 Batch Normalization， Layer Normalization 等 Trick 大大减轻了我们需要精选权重初始化方案的需要，但对于大多数情况下， 选择合适的初始化方案依旧有利于加速我们模型的收敛。

从根本上看，选择合适的初始化方案能够使得我们的损失函数便于优化（有些优化面坑坑洼洼，有些优化面比较光滑）； 从另一个角度来说， 合适的权重初始化有利于减轻梯度消失，梯度爆炸问题（参考公式推导）。



### 0x02. 初始化方案

#### 1. 常量初始化

![img](https://pic4.zhimg.com/80/v2-ca989fb91c422c566ce30b0496d0fc97_1440w.jpg)

常量初始化是错误的做法。这是因为在前向计算的时候，**所有神经元的输出均为相同**， 然后在反向传播中， **梯度相同**， **权重更新相同**，这明显是不可行的。特别地，如果初始化为0，那么梯度 = 0，参数根本无法更新。

#### 2. 随机初始化

随机初始化也是一个不好的初始化方法，因为我们不知道我们的参数会初始化为多少， 如果初始化不合理， 造成梯度消失的可能性是相当之大。另一方面，如果初始化在优化面坑坑洼洼的那一面，我们的优化过程将变得异常曲折，局部最小值，鞍点以及大的平坦区会造成优化的噩梦。

#### 3. 均匀分布初始化

```python
self.u = torch.nn.Parameter(torch.nn.init.uniform_(w, 0, 1),requires_grad=True)
```

#### 4. 高斯分布初始化

```python
self.u = torch.nn.Parameter(torch.nn.init.normal_(tensor,mean = 0.0,std = 1.0),requires_grad = True)
```

#### 5. Xavier 初始化

早期的参数初始化方法普遍是将数据和参数初始化为高斯分布（均值0方差1），但随着神经网络深度的增加，这方法并不能解决**梯度消失**问题。Xavier初始化方法来自于方法来源于2010年的一篇论文[《Understanding the difficulty of training deep feedforward neural networks》](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)。思想是：激活值的**方差**是逐层递减的，这导致反向传播中的梯度也逐层递减。要解决梯度消失，就要避免激活值方差的衰减，最好各层输入方差一致，且各层梯度的方差也一致。

文章假设的是线性激活函数，经过推导得到

​                                                          ![W  \sim U [ -\frac{\sqrt{6}}{\sqrt{n_i+ n_{i + 1}}}, \frac{\sqrt{6}}{\sqrt{n_i + n_{i + 1}}}]](https://www.zhihu.com/equation?tex=W%20%20%5Csim%20U%20%5B%20-%5Cfrac%7B%5Csqrt%7B6%7D%7D%7B%5Csqrt%7Bn_i%2B%20n_%7Bi%20%2B%201%7D%7D%7D%2C%20%5Cfrac%7B%5Csqrt%7B6%7D%7D%7B%5Csqrt%7Bn_i%20%2B%20n_%7Bi%20%2B%201%7D%7D%7D%5D)  

其中， ![n_i](https://www.zhihu.com/equation?tex=n_i)  是第i层神经元个数， ![n_{i+1}](https://www.zhihu.com/equation?tex=n_%7Bi%2B1%7D)  是第i+1层神经元个数。

推导：https://github.com/songyingxin/NLPer-Interview/blob/master/5-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E8%B0%83%E5%8F%82%20-%20%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96.md

```python
self.w_params = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),requires_grad=True)
```



#### 6. Kaiming 初始化

kaiming初始化的出现是因为xavier存在一个不成立的假设，那就是假设激活函数都是线性的，而在深度学习中常用的ReLu等都是非线性的激活函数。而kaiming初始化本质上是**高斯分布**初始化，其均值为0，方差为2/n。

​                                                                              ![W\sim N(0,\sqrt{\frac{2}{n}})](https://www.zhihu.com/equation?tex=W%5Csim%20N(0%2C%5Csqrt%7B%5Cfrac%7B2%7D%7Bn%7D%7D))  

n  为所在层的输入维度。

- Kaiming 均匀分布

  ```python
  torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
  ```

- Kaiming 正态分布

  ```python
  torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
  ```



- **a** – the negative slope of the rectifier used after this layer (only used with `'leaky_relu'`)
- **mode** – either `'fan_in'` (default) or `'fan_out'`. Choosing `'fan_in'` preserves the magnitude of the variance of the weights in the forward pass. Choosing `'fan_out'` preserves the magnitudes in the backwards pass.
- **nonlinearity** – the non-linear function (nn.functional name), recommended to use only with `'relu'` or `'leaky_relu'` (default).