# Neural Factorization Machine (NFM) [2017]:

> FM之精髓，其上在于latent embedding，有了它才能把**交互拆解**，提升泛化能力；居中在于element-wise乘，能让两个特征之间交互；其下在于**点积**，把好不容易带进来的高维信息全部压缩完了。



FM的精髓实际在于利用了隐式的embedding，但是点积实质上仍然对信息做了压缩。当我们做到element-wise这一步，其实已经把FM的优势大体上包进来了。然而如果下一步做**sum，会损失一些信息**，那么有没有办法，取FM的精髓，但又不造成信息损失呢？

### 1. NFM

出自论文 Neural Factorization Machines for Sparse Predictive Analytics 。在NFM这个方法中，作者认为**点积某种程度上限制了FM的能力上限**，因为embedding比较长，是包含更多信息的，而点积之后就**只剩下一个数了**。比如PNN，就是计算两两点积，然后输入到下一层DNN中。如果仅仅做到element-wise就停止，然后往下直接接DNN，这样就能够保存长的embedding，而不会导致信息丢失。那多是一件美事！

NFM的计算公式：

​                                                                        $$\hat{y}_{NFM}(x) = w_0+\sum_{i=1}^nw_ix_i+f(\textbf{x})$$

其中，$x_i$是one-hot向量。

下图只表示了$f(\textbf{x})$部分：

![img](https://pic2.zhimg.com/v2-22c31f89569e3d24e732e82e2cfe7fc9_b.png)



重要的模块是Bi-Interaction Pooling，用来建模二阶交叉特征。Bi-Interaction Pooling的公式：

![\begin{aligned} f_{BI}(V_x)=\sum_{i=1}^n\sum_{j=i+1}^nx_iv_i \odot x_jv_j \end{aligned} \qquad (2)](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D%20f_%7BBI%7D(V_x)%3D%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di%2B1%7D%5Enx_iv_i%20%5Codot%20x_jv_j%20%5Cend%7Baligned%7D%20%5Cqquad%20(2))

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Codot) 表示element-wise乘法，即哈达玛积，表示两个向量对应元素相乘，其结果为一个向量。所以，Bi-Interation Layer其实就是将embedding vectors 进行两两交叉  $\odot$ 运算，然后将所有向量进行对应元素求和，最终 ![f_{BI}(V_x)](https://www.zhihu.com/equation?tex=f_%7BBI%7D(V_x))为pooling之后的一个向量。

Bi-Interaction的原型还是按照原始FM来的，把所有的交叉都表示到**element-wise乘法**这一步，就停止，不再做sum了，防止信息的丢失。将长embedding pooling之后作为输入送进DNN中。我们都知道DNN具有很好的非线性，二阶信息叠加高度非线性，那我们其实是有理由展望一下**更高阶**的交叉**可能**存在。

### 2. 总结

NFM的思想：让模型在浅层尽可能包含更多的信息、减小信息丢失，来降低后续下游DNN的学习负担，这样DNN可以使用较浅的网络和较少的参数。

其实在NFM这里隐隐透露出可以做更高阶的交叉的影子了，但是为什么我们只强调是可能存在呢，因为MLP不会替你做交叉，而且这里的交叉还是不如DCN里面**形式上那么明显**，是隐式的特征交叉。NFM的主要立论点是要扩展FM的上限，强化FM的能力，其实主要目标不在高阶交叉上，而DCN这一类方法就完全是奔着高阶交叉去的了。