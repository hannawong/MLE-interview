# KNN

## 0x01. 参数化(parametric) vs. 非参数化方法

- 参数化方法: 提前假定某种拟合的函数形式

- - 优点：简单，易于解释和评估

- - 缺点：high bias, 因为真实的数据分布可能根本就不符合我们之前假定的函数形式！

- 非参数化方法:  分布的估计是数据驱动的 (Distribution or density estimate is **data-driven**)

- - Relatively **few assumptions** are made a priori about the functional form.

- - 非参数化的训练过程非常简单（可以说没有训练过程），就是把所有的训练样例存储起来。之后我们需要一个相似度函数 f，在测试过程中，计算测试样例和训练样例的相似度即可。

## 0x02. 1-近邻 (1-NN)
**【定义1】Voronoi tessellation (Dirichlet tessellation)**
当看到空间中的一系列给定的点，例如x, y1, y2, y3,…，我们希望为每个点，例如点x，划定一个包围这个点的区域(**Voronoi Cell**)。对于任意一个位于区域内的点，我们总希望它距离点x的距离小于离其他所有的给定的点，例如 y1, y2, y3,… 的距离。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220203631988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



这里提到Voronoi tessellation的意义在于说明，对欧氏空间的任何一个点x，都能够找到到它**距离最近**的那个点（有可能有多个）。

## 0x03. K-NN

1-NN的显著缺点就是，如果测试样本的最近邻(1-NN)是个**噪声**怎么办？于是想到了类似集成学习的思想，我们可以关注测试样本的**多个邻居**。如果是分类问题，让这些邻居投票来决定测试样本的类别；如果是回归问题，就对每个邻居投票的结果做加权平均。

#### 3.1 距离度量
介绍四种距离度量方法：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220204239477.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



#### 3.2 属性归一化
KNN中，将不同的属性归一化是非常重要的。因为KNN的距离函数是Minkowski距离，这个距离度量受量纲影响很大。如果某些属性的值过大可能会dominate距离函数。
比如，在判断借贷风险这个问题上，我们关注用户的年龄和年收入。年龄范围[20,70], 年收入范围[100000,600000]. 这样，显然距离函数都是由年收入决定的。
所以，要把不同属性都归一化到[0,1]区间。

#### 3.3 属性加权
可以根据不同属性的重要性给不同属性不同的权重：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220205204975.png)

例如，通过计算不同属性的**互信息**来得到这个权重。

#### 3.4 K值的选择
- 如果选择较小的K值，只有与输入实例较近或相似的训练实例才会对预测结果起作用，K值的减小就意味着整体模型变得**复杂**，容易发生过拟合；

- 如果选择较大的K值，就相当于用较大领域中的训练实例进行预测，其优点是可以减少学习的variance，但缺点是学习的bias会增大。这时候，与输入实例较远（不相似的）训练实例也会对预测器作用，使预测发生错误，且K值的增大就意味着整体的模型变得简单。推广到极限，若K=N，则完全不足取，因为此时无论输入实例是什么，都只是简单的预测它属于在训练实例中最多的类，模型过于简单，忽略了训练实例中大量有用信息。

在实际应用中，K值一般取一个比较小的数值。并采用交叉验证法（一部分样本做训练集，一部分做测试集）来选择最优的K值。

**KNN 是一种稳定的算法 – 训练样本上的一点扰动不会很大的影响最终结果。**



#### ⭐3.5 KNN优化 – KD树
既然我们要找到**k个最近的邻居**来做预测，那么我们只需要计算预测样本和所有训练集中的样本的距离，然后计算出距离最小的k个点即可，接着多数表决，很容易做出预测。这个方法的确简单直接，在样本量少、样本特征少的时候有效。但是在实际运用，我们经常碰到样本的特征数有上千以上，样本量有几十万以上，如果我们这要去预测少量的测试集样本，算法的时间效率很成问题。因此，这个方法我们一般称之为蛮力实现。
那么我们有没有其他的好办法呢？有！就是KD树实现。

scikit-learn里使用了蛮力实现(**brute-force**)，KD树实现(**KDTree**)和球树(**BallTree**)实现三种方法，这里只介绍KD树实现。

##### 3.5.1 KD树实现原理
KD树算法没有一开始就尝试对测试样本分类，而是先对训练集建模，建立的模型就是KD树，建好了模型再对测试集做预测。所谓的KD树就是K个特征维度的树。注意这里的K和KNN中的K的意思不同。KNN中的K代表最近的K个样本，KD树中的K代表样本特征的维数。为了防止混淆，后面我们称特征维数为n。

KD树算法包括三步，第一步是**建树**，第二部是**搜索**最近邻，最后一步是预测。

##### 3.5.2 KD树的建立
我们首先来看建树的方法。

从$m$个样本的$n$维特征中，分别计算$n$个特征的取值的方差，用方差最大的特征$n_k$来作为根节点。对于这个特征，我们选择特征$n_k$的取值的中位数$n_{kv}$对应的样本作为划分点，对于特征取值小于$n_{kv}$的样本，我们划入左子树，对于特征$n_k$的取值大于等于$n_{kv}$的样本，我们划入右子树。

对于左子树和右子树，我们采用和刚才同样的办法来找方差最大的特征来做根节点，**递归的生成KD树**。直到满足停止条件 – 每个分隔区域中的点数小于$m$ **OR** 分隔区域的宽度小于min_width.
具体流程如下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220211612308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



例子：

1. 最初的点集

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220211929383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



2. 在维度X上划分

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220212029669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

3. 展开左子树：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220212131925.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

4. 分别展开左、右子树后得到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220212211741.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



**3.5.3 KD树搜索最近邻**

在kd树中搜索一个节点和在二叉搜索树中极其类似。
如果当前维度的值节点小则转左，大则转右，进行下一个维度的搜索，将最终得到的叶子节点设为当前最近点。
和二叉搜索树不同，kd树的查找还需要一个递归回退的过程。从当前最近点开始，依次回退并检查兄弟节点是否存在有比当前最近点更近的点。这一操作仅需比较目标点和分离超平面的距离和当前最近的距离即可，若前者小，则有可能存在更近点，须递归访问，直到最后回退到根节点即完成了搜索。

Ex:
首先，找到测试样本(红色点)所落在的叶子节点：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220214026527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



回退：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220214325904.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220214425238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



所以，要想找测试样例的最近邻，只需关注这两个区间：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210220214559837.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



##### 3.6 KNN算法的优缺点
优点:

1） 既可以用来做分类也可以用来做回归

2） 可用于非线性分类

3） 训练时间复杂度比支持向量机之类的算法低，仅为O(n)

4） 和朴素贝叶斯之类的算法一样，对数据没有假设。

5）比较适用于样本容量比较大的自动分类，而那些样本容量较小的采用这种算法比较容易产生误分

缺点：

1）计算量大，尤其是特征数非常多的时候

2）样本不平衡的时候，对稀有类别的预测准确率低

3）模型建立需要大量的内存

4）使用懒散学习方法，基本上不学习，导致预测时速度比起逻辑回归之类的算法慢

5）相比决策树模型，KNN模型可解释性不强





----

参考资料
[1] 2020春 THU 机器学习概论课件
[2] https://www.cnblogs.com/pinard/p/6061661.html
