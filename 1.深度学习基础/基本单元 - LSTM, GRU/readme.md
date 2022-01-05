# LSTM, GRU



### 1. 【梯度消失】和【梯度爆炸】



### 1.1 梯度消失

【定义】当很多的层都用特定的激活函数(尤其是sigmoid函数)，损失函数的梯度会趋近于0，因此模型更加不容易训练。(As more layers using certain activation functions are added to neural networks, the gradients of the loss function approaches zero, making the network hard to train.)

以最简单的网络结构为例，假如有三个隐藏层，每层神经元个数都是1，且对应的非线性函数为sigmoid:

![img](https://pic2.zhimg.com/80/v2-bf9b4faafc6678b489e43c6893cb64c1_1440w.png)

每个节点的输出 ![[公式]](https://www.zhihu.com/equation?tex=y_i+%3D+%5Csigma%28w_ix_i%2Bb_i%29) , 那么

![img](https://pic4.zhimg.com/80/v2-ca989fb91c422c566ce30b0496d0fc97_1440w.jpg)

梯度消失的罪魁祸首是sigmoid函数，在sigmoid函数靠近0和1的位置，其导数很小。很多小的值相乘，导致最终的梯度很小。(自己推导： ![\sigma'(x) = \sigma(x)(1-\sigma(x))](https://www.zhihu.com/equation?tex=%5Csigma%27(x)%20%3D%20%5Csigma(x)(1-%5Csigma(x)))  )

![img](https://pic2.zhimg.com/80/v2-cc42dc6326273abbf14837d83ad805c9_1440w.jpg)                                



由于我们初始化的网络权值通常都小于1，因此当层数增多时，小于0的值不断相乘，最后就导致梯度消失的情况出现。同理，当权值过大时，导致大于1的值不断相乘，就会产生梯度爆炸。

如果一个深层网络有很多层，梯度消失导致网络只等价于后面几层的浅层网络的学习，而前面的层不怎么更新了：

![img](https://pic3.zhimg.com/80/v2-99a5b1f741226025f8a1f61f7cfdeb82_1440w.jpg)



在RNN中，也会出现梯度消失的问题，比如下面这个例子：

![img](https://pic4.zhimg.com/80/v2-c85f9d5ee35bace1599d5a9d6a0b1c73_1440w.jpg)

这里应该填"ticket",但是如果梯度非常的小，RNN模型就不能够学习在很久之前出现的词语和现在要预测的词语的关联。也就是说，RNN模型也不能把握长期的信息。

**梯度消失有几种常见的解决方法：**

（1）用下文提到的LSTM/GRU (其实这也是一种skip-connection)

（2）加上一些skip-connection, 让梯度直接流过而不经过bottleneck。例如resnet：

![img](https://pic3.zhimg.com/80/v2-a5fda1b72295d90d9cbbe2c924a69636_1440w.jpg)

（3）用Relu、Leaky relu等激活函数
**ReLu：** 让激活函数的导数为1
**LeakyReLu：** 包含了ReLu的几乎所有优点，同时解决了ReLu中0区间带来的影响

（4）使用BatchNorm/LayerNorm, 让数据分布在非饱和区、远离死区

（5）合适的权重初始化



**1.2 梯度爆炸**

回忆梯度更新的公式：

![img](https://pic1.zhimg.com/80/v2-32c4e5d9c591291d8cabe60481eba444_1440w.jpg)

那么，如果梯度太大，则参数更新的过快。步子迈的太大就会导致训练非常不稳定(训飞了)，甚至最后loss变成**Inf**。

梯度爆炸的解决方法：

（1）gradient clipping

![img](https://pic2.zhimg.com/80/v2-72e336f2d9a603e8ce7a00d0c1b23f51_1440w.jpg)如果梯度大于某个阈值了，就对其进行裁剪，让它不要高于那个阈值。

![img](https://pic4.zhimg.com/80/v2-da8a1c6fbf13d9b7a374c7276d8217b3_1440w.jpg)



(2) **权重正则化** 。如果发生梯度爆炸，那么权值的范数就会变的非常大。通过限制正则化项的大小，也可以在一定程度上限制梯度爆炸的发生。



### 2. LSTM

Vanilla RNN最致命的问题就是，它不能够保留很久之前的信息(由于梯度消失)。这是因为它的隐藏状态在不停的被重写：

![[公式]](https://www.zhihu.com/equation?tex=h%5E%7B%28t%29%7D+%3D+%5Csigma%28W_hh%5E%7B%28t-1%29%7D%2BW_ee%5E%7Bt%7D%2Bb%29)

所以，可不可以有一种RNN，能够有独立的记忆(separated memory)呢？

**2.1 LSTM 基本思想**

对于任一时间 t，都有三个概念：

- hidden state: n维向量
- cell state: n维向量，存储长期记忆。cell就像一个小小的计算机系统，可以**读、写、擦除**。
- gates: **n维向量**，每个元素的大小都是0~1之间（之后做element-wise product）。决定哪些信息可以穿过，哪些需要被挡住。

**（1）三个gate的计算**

首先，计算三个gate，它们都由上一个hidden state的输出 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7B%28t-1%29%7D) 和当前的input ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28t%29%7D) 计算得到。gate是n维向量：

![img](https://pic1.zhimg.com/80/v2-c64e0bffc32eb69ee9ef4125d33679bc_1440w.jpg)

（sigmoid函数是用于门控上最合适的激活函数）

**(2) cell 和 hidden state 的更新**

![img](https://pic2.zhimg.com/80/v2-9a14dd4e090527bb9ccee6f1365be041_1440w.jpg)

**cell**存放长期记忆，t时刻的长期记忆 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7B%28t%29%7D) 由两部分组成：①旧信息 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7B%28t-1%29%7D) 遗忘一部分；②新信息 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bc%7D%5E%7B%28t%29%7D) 写入一部分。

t时刻的**hidden state** ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7B%28t%29%7D) 就是选择一部分长期记忆 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7B%28t%29%7D) 输出的结果。

----



##### Q/A: 融合信息时为何选择 tanh？

- 值域为 (-1, 1)， 这样会带来两个好处：

  > - 与大多数情景下特征分布以 0 为中心相吻合。（激活函数一章中有提到这点特性的重要性）
  > - 可以避免前向传播时的数值溢出问题(主要是上溢)

- tanh 在 0 附近有较大的梯度，模型收敛更快



------

LSTM图示：

![img](https://pic1.zhimg.com/80/v2-ebd137e9f34ba4713e19ebd52140e660_1440w.jpg)LSTM的图示。

图中，每一个绿色方块是一个timestep。和普通的RNN一样，LSTM也是每一步有输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28t%29%7D) ，有隐藏状态 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7B%28t%29%7D) 作为输出。



**2.2 为什么LSTM能够解决梯度消失**

LSTM能够让RNN一直保留原来的信息(preserve information over many timesteps)。如果LSTM的遗忘门被设置成1，那么LSTM会一直记住原来每一步的旧信息。相比之下，RNN很难能够学习到一个[参数矩阵](https://www.zhihu.com/search?q=%E5%8F%82%E6%95%B0%E7%9F%A9%E9%98%B5&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A407089165%7D) ![[公式]](https://www.zhihu.com/equation?tex=W_h) 能够保留hidden state的全部信息。

所以，可以说LSTM解决梯度消失的主要原因是因为它有**skip-connection**的结构，能够让信息直接流过。而vanilla RNN每一步backprop都要经过 ![[公式]](https://www.zhihu.com/equation?tex=W_h) 这个bottleneck,导致梯度消失。



### 3. GRU(gated recurrent unit)

**3.1 GRU的基本思想**

跟LSTM不同的是，GRU没有cell state，只有hidden state和两个gate。

**（1）gate的计算：**

![img](https://pic4.zhimg.com/80/v2-29de427c1a94bf9ae5a524228e895a7b_1440w.jpg)

- update gate: 相当于LSTM中的forget gate(擦除旧信息)和input gate(写入新信息)
- reset gate: 判断哪一部分的hidden state是有用的，哪些是无用的。

**（2）hidden state的计算**

![img](https://pic1.zhimg.com/80/v2-87b224e944c6af7ac0bde2ffd9250c88_1440w.jpg)

**3.2 为什么GRU能解决梯度消失？**

就像LSTM一样，GRU也能够保持长期记忆(想象一下把update gate设置成0，则以前的信息全部被保留了)，也是一种增加skip-connection的方法。

**3.3 LSTM vs GRU**

- LSTM和GRU并没有明显的准确率上的区别
- GRU比起LSTM来，参数更少，运算更快，仅此而已。
- 所以，在实际应用中，我们用LSTM做default方法，如果追求更高的性能，就换成GRU



## 4. Bidirectional RNN

**4.1 单向RNN的局限性**

![img](https://pic4.zhimg.com/80/v2-ebd81b5aaba5a72830b029278bb5e57f_1440w.jpg)



**4.2 双向RNN**

![img](https://pic2.zhimg.com/80/v2-4af3f06f2b694822af50fe8a55939295_1440w.jpg)

把forward RNN和backward RNN的hidden state都拼接在一起，就可以得到包含双向信息的hidden state。这种”伪双向“的方法是Elmo的典型做法。



![img](https://pic1.zhimg.com/80/v2-a1ffc8f6e56e5045d13350ac1159899c_1440w.jpg)

【注意】只有当我们有**整句话**的时候才能用双向RNN。对于[language model](https://www.zhihu.com/search?q=language+model&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A407089165%7D)问题，就不能用双向RNN，因为只有左边的信息。



## 5. Multi-layer RNNs

多层RNN也叫 *stacked RNNs* .

**5.1 多层RNN结构**

下一层的hidden state作为上一层的输入：

![img](https://pic4.zhimg.com/80/v2-c7d1179aa11ce93d0199730f976526b3_1440w.jpg)



**5.2 多层RNN的好处**

多层RNN可以让RNN网络得到词语序列更加复杂的表示（more complex representations）

- 下面的RNN层可以得到低阶特征(lower-level features，如syntax特征)
- 上面的RNN层可以得到高阶特征(higher-level features，如semantic特征)

**5.3 多层RNN的应用**

![img](https://pic3.zhimg.com/80/v2-ddb0c586910dae579d4cdb0739c7dd8a_1440w.jpg)

【注意】如果multi-layer RNN深度很大，最好用一些skip connection



----



**Q/A：Relu 能否作为RNN的激活函数？**

RNN 的梯度消失，梯度爆炸问题在于： 

 ![\prod_{j=k+1}^t \frac{\delta S_j}{\delta S_{j-1}} = \prod_{j=k+1}^t tanh' W_h](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%20%5Cfrac%7B%5Cdelta%20S_j%7D%7B%5Cdelta%20S_%7Bj-1%7D%7D%20%3D%20%5Cprod_%7Bj%3Dk%2B1%7D%5Et%20tanh%27%20W_h)  

### 

答案是可以，但会产生一些问题：

> - 换成 Relu 可能使得输出值变得特别大，从而产生溢出
> - 换成Relu 也不能解决梯度消失，梯度爆炸问题，因为还有 $W_h$ 连乘的存在（如1中公式）

为什么 CNN 和前馈神经网络采用 Relu 就能解决梯度消失，梯度爆炸问题？

> 因为CNN 或 FFN 中各层的 W 并不相同， 且初始化时是独立同分布的，一定程度上可以抵消。
>
> 而 RNN 中各层矩阵 $W_h$ 是一样的。