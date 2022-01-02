# DCN-V2: Improved Cross&Deep Network [2020]

### 1. 相比DCN-V1的改进

DCN-V1结构图：

![img](https://pic3.zhimg.com/v2-ad567e2e785e4d58c92b0d048daa1936_b.png)

它的核心表达式为：                                  ![x_{(l+1)}=x_0x_{(l)}^Tw+b+x_{(l)}](https://www.zhihu.com/equation?tex=x_%7B(l%2B1)%7D%3Dx_0x_%7B(l)%7D%5ETw%2Bb%2Bx_%7B(l)%7D)

而V2的结构为：

![img](https://pic1.zhimg.com/v2-ad511d769a222bf1a12301942dcfc00c_b.jpg)

它的表达式可以写为：

![img](https://pic4.zhimg.com/v2-6f43b3a66f75654a1331f2eb155bcb2f_b.jpeg)

可以看出，最大的变化是将原来的向量 ![w](https://www.zhihu.com/equation?tex=w) 变成了矩阵。而这一个改动就解决了前面的问题。一个矩阵 ![W](https://www.zhihu.com/equation?tex=W) 拥有足够多的参数来保留高阶交叉信息，或者挑选需要的交叉结果。因此这个工作也实现了真正的高阶交叉。

要注意的一个DCN-V2和xDeepFM的很大区别是，DCN-V2仍然不是vector-wise的操作。根源在于，DCN-V2把所有特征的embedding concat起来一起输入网络，所以在 ![W_l](https://www.zhihu.com/equation?tex=W_l)W_l 那里无法保持同一个特征的embedding同进退，同一段embedding自己内部也存在交叉。

### 3.2 Mixture of Low-Rank DCN

![W_l \in \mathbb{R}^{d \times d}](https://www.zhihu.com/equation?tex=W_l%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20d%7D)W_l \in \mathbb{R}^{d \times d} , 参数量非常大。因此作者引入了低秩分解来处理，即把 ![W_l](https://www.zhihu.com/equation?tex=W_l)W_l 变成两个"瘦"矩阵的乘，即 ![U, V\in R^{d\times r}](https://www.zhihu.com/equation?tex=U%2C%20V%5Cin%20R%5E%7Bd%5Ctimes%20r%7D)U, V\in R^{d\times r} , ![UV^T=W](https://www.zhihu.com/equation?tex=UV%5ET%3DW)UV^T=W 。当 ![r\le d/2](https://www.zhihu.com/equation?tex=r%5Cle%20d%2F2)r\le d/2 的时候，就能够达到压缩的目的。此时有：

![x_{(l+1)}=x_0\odot (U_l(V_l^Tx_{(l)})+b_l)+x_0](https://www.zhihu.com/equation?tex=x_%7B(l%2B1)%7D%3Dx_0%5Codot%20(U_l(V_l%5ETx_%7B(l)%7D)%2Bb_l)%2Bx_0)x_{(l+1)}=x_0\odot (U_l(V_l^Tx_{(l)})+b_l)+x_0 

DCN-mix中结合了MMoE的思想，认为矩阵的低秩分解其实是在不同特征空间上的映射，所以可以采用多个特征空间，然后用门控网络进行加权求和：

![img](https://pic4.zhimg.com/v2-162e358ba4c5dc94cf25e18e21a1685b_b.jpeg)



  