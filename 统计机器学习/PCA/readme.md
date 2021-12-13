# PCA

### 0x01. 背景

在现实业务场景中，处理成千上万甚至千万量级维特征的情况也并不罕见。在这种情况下，机器学习的资源消耗是不可接受的，因此我们必须对数据进行**降维**。降维当然意味着信息的丢失，不过鉴于实际数据本身常常存在的相关性，我们可以想办法在降维的同时将信息的损失尽量降低。

那么如何降维呢？例如在做CTR预估时，一个店铺的“浏览量”和“访客数”往往具有较强的相关关系，而“下单数”和“成交数”也具有较强的相关关系。这时，如果我们删除浏览量或访客数其中一个指标，我们应该并不会丢失太多信息。因此我们可以删除一个，以降低机器学习算法的复杂度。

上面给出的是降维的朴素思想描述，可以有助于直观理解降维的动机和可行性，但并不具有操作指导意义。例如，我们到底删除哪一列损失的信息才最小？亦或，根本不是单纯删除几列，而是通过某些变换将原始数据变为更少的列但又使得丢失的信息最小？主成分分析就是这样的降维算法。

### 0x02. PCA原理详解

首先，我们先来复习一些先修知识~

#### 1. 先修知识复习

- 基：模为1的一组正交向量。可以看作一个“坐标系”。
- 样本方差：![S^{2}=\frac{1}{n-1}\sum_{i=1}^{n}{\left( x_{i}-\bar{x} \right)^2}](https://www.zhihu.com/equation?tex=S%5E%7B2%7D%3D%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft(%20x_%7Bi%7D-%5Cbar%7Bx%7D%20%5Cright)%5E2%7D)度量分散程度。
- 协方差：表示两个分布的线性相关性

​                                                   $$\begin{align*} Cov\left( X,Y \right)&=E\left[ \left( X-E\left( X \right) \right)\left( Y-E\left( Y \right) \right) \right] \\ &=\frac{1}{n-1}\sum_{i=1}^{n}{(x_{i}-\bar{x})(y_{i}-\bar{y})} \end{align*}$$

#### 2. PCA概念

PCA(principal component analysis)即主成分分析方法，是一种使用最广泛的数据降维算法。PCA的主要思想是将n维特征映射到k维上，这k维是全新的正交向量，也被称为主成分。我们的目的就是，让原先的数据在降维之后的k维正交基空间上能够分得最开(用数学的话来讲，就是方差最大)。

![img](https://pic1.zhimg.com/v2-d429d1fd00d38bce4e43453f8e24e5d0_b.png)

PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择与数据本身是密切相关的。其中，第一个新坐标轴选择是原始数据中方差最大的方向；第二个新坐标轴选取是与第一个坐标轴正交的平面中方差最大的；第三个轴是与第1,2个轴正交的平面中方差最大的。依次类推，可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，**大部分方差都包含在前面k个坐标轴中**，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面k个含有绝大部分方差的坐标轴。事实上，这相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为0的特征维度，实现对数据特征的降维处理。

思考：我们如何得到这些包含最大差异性(方差)的主成分方向(标准正交基)呢？

答案：事实上，通过计算数据矩阵的**协方差矩阵**，然后得到协方差矩阵的特征值，选择特征值最大(即方差最大)的k个特征所对应的特征向量组成的矩阵。这样就可以将数据矩阵转换到新的空间当中，实现数据特征的降维。

#### 3. PCA的一个简单实例

假设我们现在有五条数据，每个数据是二维的(第一行是x值，第二行是y值)：

![X=\left( \begin{matrix} 1 & 1 &2&4&2\\ 1&3&3&4&4 \end{matrix} \right)](https://www.zhihu.com/equation?tex=X%3D%5Cleft(%20%5Cbegin%7Bmatrix%7D%201%20%26%201%20%262%264%262%5C%5C%201%263%263%264%264%20%5Cend%7Bmatrix%7D%20%5Cright))

**（1）零均值化**

为了后续处理方便，我们首先将每行都减去该行的均值，其结果是将每个字段都变为均值为0（这样做的道理和好处后面会看到）。于是，数据变成了这样：

![X=\left( \begin{matrix} -1 & -1 &0&2&0\\ -2&0&0&1&1 \end{matrix} \right)](https://www.zhihu.com/equation?tex=X%3D%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-1%20%26%20-1%20%260%262%260%5C%5C%20-2%260%260%261%261%20%5Cend%7Bmatrix%7D%20%5Cright))

我们可以看下五条数据在平面直角坐标系内的样子：

![img](https://pic4.zhimg.com/v2-3f95215ad5ec1e50fe0a22064a39a067_b.png)

现在问题来了：如果我们必须使用一维来表示这些数据，又希望尽量保留原始的信息，你要如何选择？

通过上一节对基变换的讨论我们知道，这个问题实际上是要在二维平面中选择一个方向，将所有数据都投影到这个方向所在直线上。这是一个实际的二维降到一维的问题。那么如何选择这个方向（或者说基）才能尽量保留最多的原始信息呢？一种直观的看法是：希望投影后的投影值尽可能分散（方差最大）。

以上图为例，可以看出如果向x轴投影，那么最左边的两个点会重叠在一起，中间的两个点也会重叠在一起，这是一种严重的信息丢失。我们直观目测，如果向蓝色的那条线投影，则五个点在投影后还是可以区分的，而且能够分得最开。

下面，我们用数学方法表述这个问题。

**① 方差**

上文说到，我们希望投影后投影值尽可能分散，而这种分散程度，可以用数学上的方差来表述:

![Var(X) = \frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^2](https://www.zhihu.com/equation?tex=Var(X)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5En(x_i-%5Cbar%7Bx%7D)%5E2)

由于上面我们已经将每个字段的均值都化为0了，因此方差可以直接用每个元素的平方和除以元素个数表示：

![Var(X) = \frac{1}{n}\sum_{i=1}^nx_i^2](https://www.zhihu.com/equation?tex=Var(X)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5Enx_i%5E2)

于是上面的问题被形式化表述为：寻找一个一维基，使得所有数据变换为这个基上的坐标表示后，其平方和最大。

**② 协方差**

对于上面提到的二维降成一维的问题来说，找到那个使得方差最大(也就是平方和最大)的方向就可以了。不过对于更高维，例如三维降到一维这个问题，与之前相同，首先我们希望找到一个方向使得投影后方差最大，这样就完成了第一个方向的选择；那么下一个方向该如何选择呢？实际上，我们应该去选择与第一个基正交的方向，也就是和第一个基完全不相关。

那么，如何去衡量两个基的相关性呢？答案就是协方差。

![Cov(X,Y) = \frac{1}{m} \sum_{i=1}^m(x_i-\bar{x})(y_i-\bar{y})](https://www.zhihu.com/equation?tex=Cov(X%2CY)%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em(x_i-%5Cbar%7Bx%7D)(y_i-%5Cbar%7By%7D))

由于已经零均值化，所以现在 ![Cov(X,Y) = \frac{1}{m} \sum_{i=1}^n x_iy_i](https://www.zhihu.com/equation?tex=Cov(X%2CY)%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5En%20x_iy_i)

**③ 协方差矩阵**

到这里还看不出什么头绪，但是这里看完之后就能够把之前说的两个条件（最大方差+基之间相关性最小）联系到一起。

设原始矩阵![A=\left( \begin{matrix} x_1 & x_2& ...& x_n\\ y_1 & y_2& ...& y_n\end{matrix} \right)](https://www.zhihu.com/equation?tex=A%3D%5Cleft(%20%5Cbegin%7Bmatrix%7D%20x_1%20%26%20x_2%26%20...%26%20x_n%5C%5C%20y_1%20%26%20y_2%26%20...%26%20y_n%5Cend%7Bmatrix%7D%20%5Cright)) 则其协方差矩阵：

![img](https://pic2.zhimg.com/v2-5c80c9ee034799848207c8c876bfec21_b.png)

这个矩阵对角线上的两个元素分别是两个字段的方差，而其它元素是X和Y的协方差 -- 两者被统一到了一个矩阵！

**④ 协方差矩阵对角化**

我们发现要达到优化，等价于将协方差矩阵对角化：即除对角线外的其它元素化为0，并且在对角线上将元素按大小从上到下排列，这样我们就达到了优化目的。

由上文知道，协方差矩阵C是一个是对称矩阵，在线性代数上，实对称矩阵有一系列非常好的性质：

> 1）实对称矩阵不同特征值对应的特征向量必然正交。 
>
> 2）设特征向量λ重数为r，则必然存在r个线性无关的特征向量对应于λ，因此可以将这r个特征向量单位正交化。

由上面两条可知，一个n行n列的实对称矩阵一定可以找到n个单位正交特征向量，设这n个特征向量 ![\{e_1,e_2,...e_n\}](https://www.zhihu.com/equation?tex=%5C%7Be_1%2Ce_2%2C...e_n%5C%7D), 我们将其按列组成矩阵：

![E = [e_1,e_2,...e_n]](https://www.zhihu.com/equation?tex=E%20%3D%20%5Be_1%2Ce_2%2C...e_n%5D)

那么协方差矩阵C必然能够对角化：

![img](https://pic4.zhimg.com/v2-7966260dfa4b075a89f44ce791c4a2a7_b.png)

### 2.4 PCA算法

设有m条、n维的数据，

1）将原始数据按列组成n行m列矩阵 ![A](https://www.zhihu.com/equation?tex=A)

2）将![A](https://www.zhihu.com/equation?tex=A)的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值

3）求出协方差矩阵 ![C = \frac{1}{m}AA^T](https://www.zhihu.com/equation?tex=C%20%3D%20%5Cfrac%7B1%7D%7Bm%7DAA%5ET)，该矩阵的对角线上为方差(需要最大化之)，其他元素为协方差(需要最小化之)

4）求出协方差矩阵的特征值及对应的特征向量

5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P（4、5步相当于对协方差矩阵做有损的SVD分解）

6）B=PA即为降维到k维后的数据 （**将原始数据投影到k个特征向量组成的单位正交基坐标系中**）,在这k个特征向量的投影上，实现了方差最大化和协方差最小化。

![img](https://pic4.zhimg.com/v2-702a97e6fe697903721a2c12e71efeb3_b.png)

举个例子：

1）上文经过零均值化的矩阵![X=\left( \begin{matrix} -1 & -1 &0&2&0\\ -2&0&0&1&1 \end{matrix} \right)](https://www.zhihu.com/equation?tex=X%3D%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-1%20%26%20-1%20%260%262%260%5C%5C%20-2%260%260%261%261%20%5Cend%7Bmatrix%7D%20%5Cright))

2）求协方差矩阵

![C=\frac{1}{5}\left( \begin{matrix} -1&-1&0&2&0\\ -2&0&0&1&1 \end{matrix} \right) \left( \begin{matrix} -1&-2\\ -1&0\\ 0&0\\ 2&1\\ 0&1 \end{matrix} \right) = \left( \begin{matrix} \frac{6}{5}&\frac{4}{5}\\ \frac{4}{5}&\frac{6}{5} \end{matrix} \right)](https://www.zhihu.com/equation?tex=C%3D%5Cfrac%7B1%7D%7B5%7D%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-1%26-1%260%262%260%5C%5C%20-2%260%260%261%261%20%5Cend%7Bmatrix%7D%20%5Cright)%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-1%26-2%5C%5C%20-1%260%5C%5C%200%260%5C%5C%202%261%5C%5C%200%261%20%5Cend%7Bmatrix%7D%20%5Cright)%20%3D%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20%5Cfrac%7B6%7D%7B5%7D%26%5Cfrac%7B4%7D%7B5%7D%5C%5C%20%5Cfrac%7B4%7D%7B5%7D%26%5Cfrac%7B6%7D%7B5%7D%20%5Cend%7Bmatrix%7D%20%5Cright))

3)求协方差矩阵的特征值与特征向量

特征值为： ![\lambda_{1}=2，\lambda_{2}=\frac{2}{5}](https://www.zhihu.com/equation?tex=%5Clambda_%7B1%7D%3D2%EF%BC%8C%5Clambda_%7B2%7D%3D%5Cfrac%7B2%7D%7B5%7D)

对应的特征向量为：

​                                                                                 ![\left( \begin{matrix} 1\\ 1 \end{matrix} \right)](https://www.zhihu.com/equation?tex=%5Cleft(%20%5Cbegin%7Bmatrix%7D%201%5C%5C%201%20%5Cend%7Bmatrix%7D%20%5Cright)) ，![ \left( \begin{matrix} -1\\ 1 \end{matrix} \right)](https://www.zhihu.com/equation?tex=%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-1%5C%5C%201%20%5Cend%7Bmatrix%7D%20%5Cright)) 

标准化后的特征向量为：

​                                                                             ![ \left( \begin{matrix} \frac{1}{\sqrt{2}}\\ \frac{1}{\sqrt{2}} \end{matrix} \right)](https://www.zhihu.com/equation?tex=%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%5Cend%7Bmatrix%7D%20%5Cright))  , ![ \left( \begin{matrix} -\frac{1}{\sqrt{2}}\\ \frac{1}{\sqrt{2}} \end{matrix} \right)](https://www.zhihu.com/equation?tex=%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%5Cend%7Bmatrix%7D%20%5Cright)) 

4）由于需要降到1维，那么就取top1特征值对应特征向量![ \left( \begin{matrix} \frac{1}{\sqrt{2}}\\ \frac{1}{\sqrt{2}} \end{matrix} \right)](https://www.zhihu.com/equation?tex=%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%5Cend%7Bmatrix%7D%20%5Cright)) 作为基，现在要把原始的数据点都映射到这个基上面:

![Y=\left( \begin{matrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{matrix} \right) \left( \begin{matrix} -1 & -1& 0&2&0\\ -2&0&0&1&1 \end{matrix} \right) = \left( \begin{matrix} -\frac{3}{\sqrt{2}} & - \frac{1}{\sqrt{2}} &0&\frac{3}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{matrix} \right)](https://www.zhihu.com/equation?tex=Y%3D%5Cleft(%20%5Cbegin%7Bmatrix%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%26%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%5Cend%7Bmatrix%7D%20%5Cright)%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-1%20%26%20-1%26%200%262%260%5C%5C%20-2%260%260%261%261%20%5Cend%7Bmatrix%7D%20%5Cright)%20%3D%20%5Cleft(%20%5Cbegin%7Bmatrix%7D%20-%5Cfrac%7B3%7D%7B%5Csqrt%7B2%7D%7D%20%26%20-%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%260%26%5Cfrac%7B3%7D%7B%5Csqrt%7B2%7D%7D%20%26%20-%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%5Cend%7Bmatrix%7D%20%5Cright))

### 3. 应用

![img](https://pic3.zhimg.com/v2-4be076b58b88f4df1e62a7e4a4e365a2_b.png)

![img](https://pic1.zhimg.com/v2-3df056f231714220b70294f724ac9970_b.png)

------

参考：

Microstrong：主成分分析（PCA）原理详解zhuanlan.zhihu.com![图标](https://pic1.zhimg.com/equation_ipico.jpg)


  