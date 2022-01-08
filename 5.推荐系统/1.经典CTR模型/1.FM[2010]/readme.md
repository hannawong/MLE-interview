# FM (因子分解机) [2010]

> 1.如果说LR是复读机，那么FM可以算作是电子词典了。
> 2.**泛化**就是我没见过你，我也能懂你，但是泛化有时候和个性化有点矛盾，属于此消彼长的关系
> 3.实践中的泛化往往来源于**拆解**，没见过组成的产品，但是见过各种**零件**，就能推断出很多的信息

FM初步具备了**泛化**能力，对于**新的特征组合**有很好的推断性质，它所需要的可学习参数也小于交叉特征很多的LR。在这个DNN的时代，FM的交叉性质也没有被完全替代，还能站在时代的浪尖上。所以说，"在DNN时代，LR打不过也加入不了；FM打不过，但是它可以加入:)"



## 1. 手动特征交叉+LR

FM之前的模型都没有**自动**进行**特征交叉**，而一般依赖**人的经验**来手动构造交叉特征，比如LR。这样的缺点十分明显：

- 人工构造，效率低下

- 特征交叉难以穷尽

- 由于数据稀疏，对于训练集中没有出现的交叉特征，模型就无法学习


例如，我们来显式的构造二阶交叉特征+LR，其公式如下：

![[公式]](https://www.zhihu.com/equation?tex=w_0%2B%5Csum_%7Bi%3D1%7D%5Enw_ix_i%2B%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di%2B1%7D%5Enw_%7Bij%7Dx_ix_j)

这样有两个缺点：

一是，要学习的二阶参数 wij 太多了，O(n^2)。

二是，这些权重之间没有任何联系(泛化能力差)。比如说有两个特征，一个是性别，一个是城市。推荐一个火锅，先遇到一个样本是 男x重庆 ，结果是点击了，男x重庆这个二阶特征有权重了，下次再遇到一个女性样本，也是重庆，女x重庆的二阶特征却没有权重。可是从人的理解来说，吃不吃辣其实重庆这个特征占了很大的权重，难道我就不可以猜女x重庆也应该有一个较大的起始值才对嘛？



## 2. FM思想

所以，FM把特征的交叉做了一步**分解**，使用了隐含的embedding，所以叫做“因子分解机”。这样的分解，让模型有了**泛化能力**。

![x∈R^d](https://www.zhihu.com/equation?tex=x%E2%88%88R%5Ed) 表示d个特征，这些特征一般都是极度稀疏的。 y$是预测的label，例如"click","non-click".那么，二阶的特征交互就是：

![img](https://pic1.zhimg.com/v2-e71480993d3845d1392415a5fb0a6978_b.png)

其中， ![V∈R^{d×k}](https://www.zhihu.com/equation?tex=V%E2%88%88R%5E%7Bd%C3%97k%7D)是d个特征的embedding table， ![⟨v_i,v_j⟩](https://www.zhihu.com/equation?tex=%E2%9F%A8v_i%2Cv_j%E2%9F%A9)直接做向量内积来做交叉特征的系数（一个数），最后得到 ![n(n-1)/2](https://www.zhihu.com/equation?tex=n(n-1)%2F2)个二阶交叉特征的值，将它们相加就得到了最后一项 ![\sum_{i=1}^d\sum_{j=i+1}^d<v_i,v_j>x_i x_j](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5Ed%5Csum_%7Bj%3Di%2B1%7D%5Ed%3Cv_i%2Cv_j%3Ex_i%20x_j)  , 这是一个数。最后，将一阶特征输入到LR中，再加上二阶交叉项，得到最后的输出logit。整个模型的示意图如下所示：

![img](https://pic3.zhimg.com/v2-fdcd14d4434853b4f96adf9996fe52da_b.png)

红色代表一阶特征+LR；蓝色表示二阶特征。



## 3. FM复杂度

FM有一个复杂度问题，也是这篇文章的一个卖点，经常出现在面试中。

然而，如果只用上述这种暴力方法来计算二阶特征交互，其复杂度为 ![O(kd^2)](https://www.zhihu.com/equation?tex=O(kd%5E2)) ,其中 $d$为embedding size. 可以用下面的方法降为 ![O(kd)](https://www.zhihu.com/equation?tex=O(kd)) 。此过程要求会推导。

![img](https://pic3.zhimg.com/v2-78a8e0068d010da00aff0d594c762c82_b.png)



空间上，假如有两个field，它们的特征取值个数分别为n1,n2，那么按照原来的方法就是要求n1,n2个参数；用FM进行embedding的方法，只需求 (n1+n2)d 个参数。



## 4. 总结

**优点：** 对训练集中未出现的交叉特征也可以进行泛化（embedding table）。

**缺点：** 只考虑了二阶交叉特征，没有考虑**更高阶**的交叉特征。



## 5. 代码实现

来源：Deepctr项目

二阶交叉项：

```python
concated_embeds_value = inputs  ###输入的特征[batchsize, 26(feat_num), 4(embed_size)]

square_of_sum = tf.square(reduce_sum(concated_embeds_value, axis=1, keep_dims=True))#(?, 1, 4)
sum_of_square = reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)##(?, 1, 4)
cross_term = square_of_sum - sum_of_square##(?, 1, 4)
cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)##(?, 1)
```

一阶交叉项和常数项直接交给LR去做即可。
