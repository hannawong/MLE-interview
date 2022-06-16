# SVM



## 1. 线性支持向量机 (Linear Support Vector Machine)

### 1.1 Max margin linear classifier

给定一组训练集 ![\{(x_1,y_1),(x_2,y_2),...(x_n,y_n)\}](https://www.zhihu.com/equation?tex=%5C%7B(x_1%2Cy_1)%2C(x_2%2Cy_2)%2C...(x_n%2Cy_n)%5C%7D), 找到分类超平面 ![w_1x_1+w_2x_2+...w_nx_n + b = 0](https://www.zhihu.com/equation?tex=w_1x_1%2Bw_2x_2%2B...w_nx_n%20%2B%20b%20%3D%200)，在该平面上面的被分为正例，在该平面下面的被分为负例。

如果给的训练集是线性可分，那么有无穷多种方法都可以进行分类：

![img](https://pic3.zhimg.com/v2-3d3574bd5d10e2e18bfd708e6bfb4b0a_b.png)

在这无穷多个分类超平面中，哪个是最好的呢？这就引入了**margin**的概念。

*什么是margin？*

![img](https://pic1.zhimg.com/80/v2-444eb631832101fe107ffa22ccd752e6_1440w.png)

引入跟分类超平面平行的两个超平面：![w_1x_1+w_2x_2+...w_nx_n + b = \pm1](https://www.zhihu.com/equation?tex=w_1x_1%2Bw_2x_2%2B...w_nx_n%20%2B%20b%20%3D%20%5Cpm1)。这两个超平面的距离 ![\frac{2}{||w||_2}](https://www.zhihu.com/equation?tex=%5Cfrac%7B2%7D%7B%7C%7Cw%7C%7C_2%7D) 就是margin。（分母是二范数）

![img](https://pic4.zhimg.com/v2-6ad2f9da0310bb797d324533f600829b_b.png)

因此，问题可以用语言表达为：**在所有点都被正确分类的前提下，最大化margin**。用数学形式表示如下:

![img](https://pic4.zhimg.com/v2-d0dba91b9e3ce73bb730d09ae540c693_b.png)

第一行最大化margin；第二行保证所有点都被正确分类



### 1.2 对偶问题求解 (Dual Problem)

如上的数学形式表达，该如何解决这样的优化问题呢？

定义拉格朗日函数，**为每一个约束条件加上一个拉格朗日乘子**(Lagrange multiplier) ![\alpha](https://www.zhihu.com/equation?tex=%5Calpha) ，将约束条件融合到目标函数里去，从而只用一个函数表达式便能清楚的表达出我们的问题：

![img](https://pic4.zhimg.com/v2-07056725a89d4a64e32aefab74b3e063_b.png)

其中， ![\alpha_i \geq 0](https://www.zhihu.com/equation?tex=%5Calpha_i%20%5Cgeq%200).

解这个问题的原始方法：

- 首先固定住w,b, 只改变 ![\alpha](https://www.zhihu.com/equation?tex=%5Calpha) , 去最大化 ![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha)). 容易看出，
- - 当某个约束条件不满足时，会有某个 ![y_i(w^Tx_i+b) -1<0](https://www.zhihu.com/equation?tex=y_i(w%5ETx_i%2Bb)%20-1%3C0) , 此时只需要让 ![\alpha_i](https://www.zhihu.com/equation?tex=%5Calpha_i)无限大，就可以最大化![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha)), 其最大值为+inf, 下一步无法最小化了。
  - 当所有约束条件都满足时，所有![y_i(w^Tx_i+b)](https://www.zhihu.com/equation?tex=y_i(w%5ETx_i%2Bb))都>0, 此时只需要让 ![\alpha_i](https://www.zhihu.com/equation?tex=%5Calpha_i)=0, 就可以最大化![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha))，其最大值为 ![\frac{1}{2}||w||^2](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E2) , 就是我们下一步要最小化的东西。
- 下一步改变w,b, 去最小化 ![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha))。假如上一步的约束条件都满足，现在就是要最小化![\frac{1}{2}||w||^2](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E2).

如果直接求解，那么一上来便得面对w和b两个参数，而 ![\alpha_i](https://www.zhihu.com/equation?tex=%5Calpha_i)又是不等式约束，这个求解过程不好做。不妨把最小和最大的位置交换一下，交换以后的新问题是原始问题的对偶问题。在满足某些条件的情况下，这两者相等，这个时候就可以通过求解对偶问题来间接地求解原始问题。

对偶问题求解：

- 首先固定 ![\alpha](https://www.zhihu.com/equation?tex=%5Calpha) , 只改变 w,b使![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha))最小化。这就需要分别对w，b求偏导数，即令 ∂L/∂w 和 ∂L/∂b 等于零：

![img](https://pic3.zhimg.com/v2-84ba37f09e93f62ab23693a5f716574e_b.png)

将以上结果代入之前的 ![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha))：

![img](https://pic4.zhimg.com/v2-07056725a89d4a64e32aefab74b3e063_b.png)

得到：

![img](https://pic2.zhimg.com/v2-1d46b7c483bd441b97e3e5cc04667cd9_b.png)

- 之后改变 ![\alpha_i](https://www.zhihu.com/equation?tex=%5Calpha_i)使得 ![L(w,b,\alpha)](https://www.zhihu.com/equation?tex=L(w%2Cb%2C%5Calpha)) 最大化，问题变成：

![img](https://pic2.zhimg.com/v2-a809a0e756f65edd07ab68205f23a849_b.png)

subject to 

![img](https://pic2.zhimg.com/v2-2cb13f9edc0c78d4bfe1251a5465edc9_b.png)



到底什么是支持向量？

![img](https://pic2.zhimg.com/v2-748de1790623045206a6f7c2cfd1c629_b.png)

SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。



### 1.3 线性不可分的情况

在线性不可分的情况下，总会有一些点被错误的分类。于是，我们修改损失函数如下，即加入了对错误分类的惩罚：

![img](https://pic4.zhimg.com/v2-5f87480a6eec0d8ee367675ac4125e1b_b.png)

这里可以看到， ![\frac{1}{2}||w||^2](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E2) 实际上就是正则化项，后面的loss才是实际的经验损失函数。这和逻辑回归的损失函数十分相似：

![img](https://pic2.zhimg.com/v2-b56d43630bc3734d17c6b56275cf9585_b.jpeg)

在SVM中，常见的损失函数有0/1loss和hinge loss：

![img](https://pic1.zhimg.com/v2-d9b00906010e0880bd963742f71c75b4_b.jpeg)

0/1 loss 只有对完全分类错误的点( ![y_i(w^Tx_i+b)<0](https://www.zhihu.com/equation?tex=y_i(w%5ETx_i%2Bb)%3C0))给惩罚，hinge loss对落在分类超平面外的点( ![y_i(w^Tx_i+b)<1](https://www.zhihu.com/equation?tex=y_i(w%5ETx_i%2Bb)%3C1))都给惩罚。

引入松弛变量 ![\xi_i>0](https://www.zhihu.com/equation?tex=%5Cxi_i%3E0):

![img](https://pic4.zhimg.com/v2-3a8c6662760cd46b7b02cc07c0fa043f_b.png)

损失函数最大化几何间隔，同时最小化松弛变量之和。现在，我们允许一些训练集落在两个分类平面之间 ![(0<\xi_i<1)](https://www.zhihu.com/equation?tex=(0%3C%5Cxi_i%3C1)), 甚至允许一些训练样本被错误分类（ ![\xi_i>1](https://www.zhihu.com/equation?tex=%5Cxi_i%3E1)）.

![img](https://pic4.zhimg.com/v2-fbcafa53d615d8cb5eafe8d5050aa5c7_b.png)

、

用上文介绍的损失函数hinge loss替换掉 ![\xi_i](https://www.zhihu.com/equation?tex=%5Cxi_i) :

![img](https://pic4.zhimg.com/v2-ebb3afe15f673affe4df0e227457b293_b.jpeg)



![img](https://pic4.zhimg.com/v2-3e6480057a577b94b3683180dca0956b_b.jpeg)

线性不可分情况的对偶问题

![img](https://pic4.zhimg.com/v2-73b4f1eca05bb0d1ae99831fc3edd757_b.png)

![img](https://pic1.zhimg.com/v2-c49f260d2e34a7cb7fff206dd8c20568_b.png)

这里，支持向量( ![\alpha_i>0](https://www.zhihu.com/equation?tex=%5Calpha_i%3E0) 的那些)包括落在边界上的，和落在两个分类边界之间的，以及分类错误的。

用对偶方法计算得到的预测值：

![img](https://pic1.zhimg.com/v2-e67f1d4f6a37b5ad0bff8802ec2639b4_b.jpeg)

## 2. Kernel Support Vector Machine

### 2.1 特征空间的隐式映射：核函数

在线性不可分的情况下，支持向量机首先在低维空间中完成计算，然后通过核函数将输入空间映射到高维特征空间，最终在高维特征空间中构造出最优分离超平面，从而把平面上本身不好分的非线性数据分开。如图所示，一堆数据在二维空间无法划分，从而映射到三维空间里划分：

![img](https://pic3.zhimg.com/v2-6e372f4829eb3ce5ecd44da54d9ac46a_b.png)

### 2.2 核函数及其构造

计算两个向量在隐式映射过后的空间中的内积的函数叫做核函数。即：将 ![<x_i,x_j>](https://www.zhihu.com/equation?tex=%3Cx_i%2Cx_j%3E)<x_i,x_j> 变为 ![<\phi(x),\phi(y)>](https://www.zhihu.com/equation?tex=%3C%5Cphi(x)%2C%5Cphi(y)%3E)<\phi(x),\phi(y)> 

![img](https://pic4.zhimg.com/v2-805c6fc912aca75c54ea2d2400dcb323_b.png)

如果有一种函数 ![k(x,y)](https://www.zhihu.com/equation?tex=k(x%2Cy))k(x,y) 能够直接计算出来![<\phi(x),\phi(y)>](https://www.zhihu.com/equation?tex=%3C%5Cphi(x)%2C%5Cphi(y)%3E)<\phi(x),\phi(y)>，这样我们就不用显式地定义 ![\phi(x)](https://www.zhihu.com/equation?tex=%5Cphi(x))\phi(x) . 这样的函数就叫做核函数。常见的核函数：

![img](https://pic1.zhimg.com/v2-ef14e910be235e52ad1d98a610ec06d0_b.jpeg)

其中，高斯核会将原始空间映射为无穷维空间。不过，如果参数选得很大的话，高次特征上的权重实际上衰减得非常快，所以实际上（数值上近似一下）相当于一个低维的子空间；反过来，如果参数选得很小，则可以将任意的数据映射为线性可分——当然，这并不一定是好事，因为随之而来的可能是非常严重的过拟合问题。不过，总的来说，通过调控参数，高斯核实际上具有相当高的灵活性，也是使用最广泛的核函数之一。下图所示的例子便是把低维线性不可分的数据通过高斯核函数映射到了高维空间：

![img](https://pic2.zhimg.com/v2-6b1d6f45795767ae6b07395bfa66e819_b.png)

------

补充：关于SVM的题目

1. 关于支持向量机SVM,下列说法错误的是（C） 

  A.L2正则项，作用是最大化分类间隔，使得分类器拥有更强的泛化能力

  B.Hinge 损失函数，作用是最小化经验分类错误

  C.分类间隔为1/||w||，||w||代表向量的模

  D.当参数C越小时，分类间隔越大，分类错误越多，趋于欠学习