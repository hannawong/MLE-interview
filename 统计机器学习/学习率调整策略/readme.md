# 学习率调整策略

### 0x01. 学习率衰减

一般来说，我们希望在训练初期学习率大一些，使得网络收敛迅速，在训练后期学习率小一些，使得网络更好的收敛到最优解。

**1、指数衰减**

学习率按照指数的形式衰减是比较常用的策略，我们首先需要确定需要针对哪个优化器执行学习率动态调整策略，也就是首先*定义一个优化器*：

```python
optimizer_ExpLR = torch.optim.SGD(net.parameters(), lr=0.1)
```

定义好优化器以后，就可以给这个优化器绑定一个指数衰减学习率控制器：

```python
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer_ExpLR, gamma=0.98)
```

其中参数gamma表示衰减的底数，选择不同的gamma值可以获得幅度不同的衰减曲线，如下：

![img](https://pic3.zhimg.com/80/v2-d990582cda2fc2aa88ae91d5aa17a6b6_1440w.jpg)



**2、固定步长衰减**

有时我们希望学习率每隔一定步数（或者epoch）就减少为原来的gamma分之一，使用固定步长衰减依旧先定义优化器，再给优化器绑定StepLR对象：

```python
optimizer_StepLR = torch.optim.SGD(net.parameters(), lr=0.1)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer_StepLR, step_size=step_size, gamma=0.65)
```

其中gamma参数表示衰减的程度，step_size参数表示每隔多少个step进行一次学习率调整，下面对比了不同gamma值下的学习率变化情况：

![img](https://pic1.zhimg.com/80/v2-a1c38e6c8e26ad3e953d1ebb67d7243c_1440w.jpg)



**3、余弦退火衰减**

严格的说，余弦退火策略不应该算是学习率衰减策略，因为它使得学习率按照周期变化，其定义方式如下：

```python
optimizer_CosineLR = torch.optim.SGD(net.parameters(), lr=0.1)
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_CosineLR, T_max=150, eta_min=0)
```

参数T_max表示余弦函数周期；eta_min表示学习率的最小值，默认它是0表示学习率至少为正值。下图展示了不同周期下的余弦学习率更新曲线：

![img](https://pic2.zhimg.com/80/v2-bb255df05eb665cc6530845bde637bc9_1440w.jpg)



**4、学习率动态更新策略的说明**

负责学习率调整的类：StepLR、ExponentialLR和CosineAnnealingLR，其完整对学习率的更新都是在其**step()函数被调用以后完成的**。根据pytorch官网上给出的说明，scheduler.step()函数的调用应该在训练代码以后：

```python
scheduler = ...
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```



### 0x02. WarmUp

warmup 需要在训练最初使用较小的学习率来启动，并很快切换到大学习率而后进行常见的衰减decay。

这是因为，刚开始模型对数据的“分布”理解为零，或者是说“均匀分布”（当然这取决于你的初始化）；在第一轮训练的时候，每个数据点对模型来说都是新的，模型会**很快地进行数据分布修正**，如果这时候学习率就很大，极有可能导致开始的时候就对该数据过拟合，后面要通过多轮训练才能拉回来，浪费时间。当训练了一段时间（比如两轮、三轮）后，模型已经对每个数据点看过几遍了，或者说对当前的batch而言有了一些正确的[先验](https://www.zhihu.com/search?q=%E5%85%88%E9%AA%8C&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A438851458%7D)，较大的学习率就不那么容易会使模型学偏，所以可以适当调大学习率。这个过程就可以看做是warmup。那么为什么之后还要decay呢？当模型训到一定阶段后（比如十个epoch），模型的分布就已经比较固定了，或者说能学到的新东西就比较少了。如果还沿用较大的学习率，就会破坏这种稳定性，用我们通常的话说，就是已经接近loss的local optimal了，为了靠近这个point，我们就要慢慢来。



BERT的预训练过程就是用了学习率WarmUp的方法。