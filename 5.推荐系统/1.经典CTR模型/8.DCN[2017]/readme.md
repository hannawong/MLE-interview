# Deep & Cross [2017] -- 高阶交叉空许约

出自论文 *Deep & Cross Network for Ad Click Predictions*。该模型主要特点在于提出Cross network，用于高阶特征的自动化显式交叉编码。这是因为传统DNN对于高阶特征的提取效率并不高，而Cross Network通过调整结构层数能够**显式构造出有限阶（bounded-degree）交叉特征**，可以提高了模型的表征能力。同时，DCN引入了残差结构的思想，使得模型能够更深。



## 0x01. 模型结构

![img](https://pic1.zhimg.com/80/v2-fa9b18644a4fe4dba9fd16c4faef137c_1440w.jpg)



还是分为两路的结构，其中Deep端就是一个普通的MLP，不必多言；Cross端是模型的核心，数学表达式如下：                  

​                                          ![\begin{aligned} X_{l+1}=X_0X_{l}^TW_{l}+b_{l}+X_{l}=f(X_{l},W_{l},b_{l})+X_{l} \end{aligned}](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D%20X_%7Bl%2B1%7D%3DX_0X_%7Bl%7D%5ETW_%7Bl%7D%2Bb_%7Bl%7D%2BX_%7Bl%7D%3Df(X_%7Bl%7D%2CW_%7Bl%7D%2Cb_%7Bl%7D)%2BX_%7Bl%7D%20%5Cend%7Baligned%7D)  

其中  ![X_{l},X_{l+1} \in \mathbb{R}^d](https://www.zhihu.com/equation?tex=X_%7Bl%7D%2CX_%7Bl%2B1%7D%20%5Cin%20%5Cmathbb%7BR%7D%5Ed)   分别代表Cross Network的第 ![l,l+1](https://www.zhihu.com/equation?tex=l%2Cl%2B1) 层的输出， ![W_{l},b_{l} \in \mathbb{R}^d](https://www.zhihu.com/equation?tex=W_%7Bl%7D%2Cb_%7Bl%7D%20%5Cin%20%5Cmathbb%7BR%7D%5Ed)  分别为该层的参数与偏置项，是需要学习的参数。x0为一开始输入特征的embedding的拼接，它会在每一层都参与运算。再使用ResNet的思想，将 ![X_l](https://www.zhihu.com/equation?tex=X_l)  直接输入下一层，可以将原始信息在CrossNet中进行传递。结构上可以用下面的图来辅助理解：

![img](https://pic2.zhimg.com/80/v2-81244edb73b8fa6b0c738cac2767eb7d_1440w.jpg)

利用 ![x_0 ](https://www.zhihu.com/equation?tex=x_0%20)  与 ![x^{l}](https://www.zhihu.com/equation?tex=x%5E%7Bl%7D)  **向量外积**得到embedding中所有的**元素**的交叉组合（注意不是每个feature的交叉组合，而是embedding元素的交叉组合，这也就是DCN中所谓"交叉"和FM中"交叉"的不同之处！），层层叠加之后便可得到任意有界阶组合特征，当cross layer叠加到 l 层，交叉最高阶可以达到 l+1 阶。下面详细地计算一下：

令 ![X_0=\left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right]](https://www.zhihu.com/equation?tex=X_0%3D%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D)那么

![\begin{aligned} X_1={} & X_0X_0^{\prime}W_0+X_0 \\ ={} &  \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \left[x_{0,1}x_{0,2}\right] \left[\begin{matrix}w_{0,1}\\w_{0,2}\end{matrix}\right] +  \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix}x_{0,1}^2,x_{0,1}x_{0,2}\\x_{0,2}x_{0,1},x_{0,2}^2\end{matrix}\right] \left[\begin{matrix}w_{0,1}\\w_{0,2}\end{matrix}\right] + \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix}w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}\\ w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2\end{matrix}\right] + \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix}w_{0,1}{\color{Red}{x_{0,1}^2}}+w_{0,2}{\color{Red}{x_{0,1}x_{0,2}}}+{\color{Red}{x_{0,1}}}\\ w_{0,1}{\color{Red}{x_{0,2}x_{0,1}}}+w_{0,2}{\color{Red}{x_{0,2}^2}}+{\color{Red}{x_{0,2}}}\end{matrix}\right] \end{aligned} \qquad (3) \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D%20X_1%3D%7B%7D%20%26%20X_0X_0%5E%7B%5Cprime%7DW_0%2BX_0%20%5C%5C%20%3D%7B%7D%20%26%20%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5Cleft%5Bx_%7B0%2C1%7Dx_%7B0%2C2%7D%5Cright%5D%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7D%5C%5Cw_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%2B%20%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%3D%7B%7D%20%26%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5E2%2Cx_%7B0%2C1%7Dx_%7B0%2C2%7D%5C%5Cx_%7B0%2C2%7Dx_%7B0%2C1%7D%2Cx_%7B0%2C2%7D%5E2%5Cend%7Bmatrix%7D%5Cright%5D%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7D%5C%5Cw_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%2B%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%3D%7B%7D%20%26%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7Dx_%7B0%2C1%7D%5E2%2Bw_%7B0%2C2%7Dx_%7B0%2C1%7Dx_%7B0%2C2%7D%5C%5C%20w_%7B0%2C1%7Dx_%7B0%2C2%7Dx_%7B0%2C1%7D%2Bw_%7B0%2C2%7Dx_%7B0%2C2%7D%5E2%5Cend%7Bmatrix%7D%5Cright%5D%20%2B%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%3D%7B%7D%20%26%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%5E2%7D%7D%2Bw_%7B0%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7Dx_%7B0%2C2%7D%7D%7D%2B%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%7D%7D%5C%5C%20w_%7B0%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7Dx_%7B0%2C1%7D%7D%7D%2Bw_%7B0%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%5E2%7D%7D%2B%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%7D%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5Cend%7Baligned%7D%20%5Cqquad%20(3)%20%5C%5C) 

继续计算 ![X_2](https://www.zhihu.com/equation?tex=X_2)，有：

![\begin{aligned} X_2={} & X_0X_1^{\prime}W_1+X_1 \\ ={} & \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right] \left[w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}+x_{0,1}, \quad  w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2+x_{0,2}\right] \left[\begin{matrix}w_{1,1}\\w_{1,2}\end{matrix}\right] \\  +{} &  \left[\begin{matrix}w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}+x_{0,1} \\ w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2+x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix} w_{0,1}x_{0,1}^3+w_{0,2}x_{0,1}^2x_{0,2}+x_{0,1}^2, \quad  w_{0,1}x_{0,2}x_{0,1}^2+w_{0,2}x_{0,2}^2x_{0,1}+x_{0,2}x_{0,1} \\ w_{0,1}x_{0,1}^2x_{0,2}+w_{0,2}x_{0,1}x_{0,2}^2+x_{0,1}x_{0,2}, \quad  w_{0,1}x_{0,2}^2x_{0,1}+w_{0,2}x_{0,2}^3+x_{0,2}^2 \end{matrix}\right] \left[\begin{matrix}w_{1,1}\\w_{1,2}\end{matrix}\right] \\ +{} &  \left[\begin{matrix}w_{0,1}x_{0,1}^2+w_{0,2}x_{0,1}x_{0,2}+x_{0,1} \\ w_{0,1}x_{0,2}x_{0,1}+w_{0,2}x_{0,2}^2+x_{0,2}\end{matrix}\right] \\ ={} & \left[\begin{matrix} w_{0,1}w_{1,1}{\color{Red}{x_{0,1}^3}}+w_{0,2}w_{1,1}{\color{Red}{x_{0,1}^2x_{0,2}}}+w_{1,1}{\color{Red}{x_{0,1}^2}} +  w_{0,1}w_{1,2}{\color{Red}{x_{0,2}x_{0,1}^2}}+w_{0,2}w_{1,2}{\color{Red}{x_{0,2}^2x_{0,1}}}+w_{1,2}{\color{Red}{x_{0,2}x_{0,1}}} \\ w_{0,1}w_{1,1}{\color{Red}{x_{0,1}^2x_{0,2}}}+w_{0,2}w_{1,1}{\color{Red}{x_{0,1}x_{0,2}^2}}+w_{1,1}{\color{Red}{x_{0,1}x_{0,2}}} +  w_{0,1}w_{1,2}{\color{Red}{x_{0,2}^2x_{0,1}}}+w_{0,2}w_{1,2}{\color{Red}{x_{0,2}^3}}+w_{1,2}{\color{Red}{x_{0,2}^2}} \end{matrix}\right] \\ +{} &  \left[\begin{matrix}w_{0,1}{\color{Red}{x_{0,1}^2}}+w_{0,2}{\color{Red}{x_{0,1}x_{0,2}}}+{\color{Red}{x_{0,1}}} \\ w_{0,1}{\color{Red}{x_{0,2}x_{0,1}}}+w_{0,2}{\color{Red}{x_{0,2}^2}}+{\color{Red}{x_{0,2}}}\end{matrix}\right] \\ \end{aligned} (4) \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D%20X_2%3D%7B%7D%20%26%20X_0X_1%5E%7B%5Cprime%7DW_1%2BX_1%20%5C%5C%20%3D%7B%7D%20%26%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5Cleft%5Bw_%7B0%2C1%7Dx_%7B0%2C1%7D%5E2%2Bw_%7B0%2C2%7Dx_%7B0%2C1%7Dx_%7B0%2C2%7D%2Bx_%7B0%2C1%7D%2C%20%5Cquad%20%20w_%7B0%2C1%7Dx_%7B0%2C2%7Dx_%7B0%2C1%7D%2Bw_%7B0%2C2%7Dx_%7B0%2C2%7D%5E2%2Bx_%7B0%2C2%7D%5Cright%5D%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B1%2C1%7D%5C%5Cw_%7B1%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%20%2B%7B%7D%20%26%20%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7Dx_%7B0%2C1%7D%5E2%2Bw_%7B0%2C2%7Dx_%7B0%2C1%7Dx_%7B0%2C2%7D%2Bx_%7B0%2C1%7D%20%5C%5C%20w_%7B0%2C1%7Dx_%7B0%2C2%7Dx_%7B0%2C1%7D%2Bw_%7B0%2C2%7Dx_%7B0%2C2%7D%5E2%2Bx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%3D%7B%7D%20%26%20%5Cleft%5B%5Cbegin%7Bmatrix%7D%20w_%7B0%2C1%7Dx_%7B0%2C1%7D%5E3%2Bw_%7B0%2C2%7Dx_%7B0%2C1%7D%5E2x_%7B0%2C2%7D%2Bx_%7B0%2C1%7D%5E2%2C%20%5Cquad%20%20w_%7B0%2C1%7Dx_%7B0%2C2%7Dx_%7B0%2C1%7D%5E2%2Bw_%7B0%2C2%7Dx_%7B0%2C2%7D%5E2x_%7B0%2C1%7D%2Bx_%7B0%2C2%7Dx_%7B0%2C1%7D%20%5C%5C%20w_%7B0%2C1%7Dx_%7B0%2C1%7D%5E2x_%7B0%2C2%7D%2Bw_%7B0%2C2%7Dx_%7B0%2C1%7Dx_%7B0%2C2%7D%5E2%2Bx_%7B0%2C1%7Dx_%7B0%2C2%7D%2C%20%5Cquad%20%20w_%7B0%2C1%7Dx_%7B0%2C2%7D%5E2x_%7B0%2C1%7D%2Bw_%7B0%2C2%7Dx_%7B0%2C2%7D%5E3%2Bx_%7B0%2C2%7D%5E2%20%5Cend%7Bmatrix%7D%5Cright%5D%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B1%2C1%7D%5C%5Cw_%7B1%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%2B%7B%7D%20%26%20%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7Dx_%7B0%2C1%7D%5E2%2Bw_%7B0%2C2%7Dx_%7B0%2C1%7Dx_%7B0%2C2%7D%2Bx_%7B0%2C1%7D%20%5C%5C%20w_%7B0%2C1%7Dx_%7B0%2C2%7Dx_%7B0%2C1%7D%2Bw_%7B0%2C2%7Dx_%7B0%2C2%7D%5E2%2Bx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%3D%7B%7D%20%26%20%5Cleft%5B%5Cbegin%7Bmatrix%7D%20w_%7B0%2C1%7Dw_%7B1%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%5E3%7D%7D%2Bw_%7B0%2C2%7Dw_%7B1%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%5E2x_%7B0%2C2%7D%7D%7D%2Bw_%7B1%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%5E2%7D%7D%20%2B%20%20w_%7B0%2C1%7Dw_%7B1%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7Dx_%7B0%2C1%7D%5E2%7D%7D%2Bw_%7B0%2C2%7Dw_%7B1%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%5E2x_%7B0%2C1%7D%7D%7D%2Bw_%7B1%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7Dx_%7B0%2C1%7D%7D%7D%20%5C%5C%20w_%7B0%2C1%7Dw_%7B1%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%5E2x_%7B0%2C2%7D%7D%7D%2Bw_%7B0%2C2%7Dw_%7B1%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7Dx_%7B0%2C2%7D%5E2%7D%7D%2Bw_%7B1%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7Dx_%7B0%2C2%7D%7D%7D%20%2B%20%20w_%7B0%2C1%7Dw_%7B1%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%5E2x_%7B0%2C1%7D%7D%7D%2Bw_%7B0%2C2%7Dw_%7B1%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%5E3%7D%7D%2Bw_%7B1%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%5E2%7D%7D%20%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%2B%7B%7D%20%26%20%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dw_%7B0%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%5E2%7D%7D%2Bw_%7B0%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7Dx_%7B0%2C2%7D%7D%7D%2B%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C1%7D%7D%7D%20%5C%5C%20w_%7B0%2C1%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7Dx_%7B0%2C1%7D%7D%7D%2Bw_%7B0%2C2%7D%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%5E2%7D%7D%2B%7B%5Ccolor%7BRed%7D%7Bx_%7B0%2C2%7D%7D%7D%5Cend%7Bmatrix%7D%5Cright%5D%20%5C%5C%20%5Cend%7Baligned%7D%20(4)%20%5C%5C)

由以上公式可知，当cross layer叠加$l$层时，"交叉"最高阶可以达到$l+1 $阶，并且包含了所有的元素交叉组合。



## 0x02. DCN真的做了特征交叉吗？

到现在为止，一切看起来都很美好。但是转念一想，高阶特征交叉明明是一个指数级的操作，而DCN用简单的矩阵乘就"做到了"，这会不会太过廉价了？

考虑只有一层的DCN，我们能让它去还原出FM的形式吗？

还是令 ![X_0=\left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right]](https://www.zhihu.com/equation?tex=X_0%3D%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D)，那么

 ![x_0x_0^Tw = \left[\begin{matrix}x_{0,1}\\x_{0,2}\end{matrix}\right]\left[\begin{matrix}x_{0,1},x_{0,2}\end{matrix}\right] w = \left[\begin{matrix}x_{0,1}^2, x_{0,1}x_{0,2} \\x_{0,1}x_{0,2}, x_{0,2}^2\end{matrix}\right]w](https://www.zhihu.com/equation?tex=x_0x_0%5ETw%20%3D%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%2Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D%20w%20%3D%20%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5E2%2C%20x_%7B0%2C1%7Dx_%7B0%2C2%7D%20%5C%5Cx_%7B0%2C1%7Dx_%7B0%2C2%7D%2C%20x_%7B0%2C2%7D%5E2%5Cend%7Bmatrix%7D%5Cright%5Dw)  

要是按照FM中的交叉特征定义（即内积），必须拿出上面 ![[公式]](https://www.zhihu.com/equation?tex=x_0x_0%5ET) 矩阵的上三角或者下三角（不包含对角线）的所有元素加到一起才可以，这时候怎么解出 ![[公式]](https://www.zhihu.com/equation?tex=w) 呢？我发现不论如何设计 ![[公式]](https://www.zhihu.com/equation?tex=w) 其实都是做不到的（本质原因还是因为$w$是个向量，势单力薄)。所以，这里的交叉最后和我们见到的FM以及类似模型中的交叉已经不是一个东西了!

当交叉变成了embedding中**元素的乘法**，而不是原来**整个embedding合起来内积**，就不存在embedding泛化性的保证了，那么交叉的意义还剩下多大呢？



对于上面的问题，xDeepFM中相当于是做了一个归纳：**DCN的本质实际上是给 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 乘了一个系数**！

结合上面的图， ![x^{l}](https://www.zhihu.com/equation?tex=x%5E%7Bl%7D)  和w乘起来就是一个数字，也就是说，最后一层一层迭代完了，只得到一个 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 的倍数。你要说没有交叉吧，系数其实还是和 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 有关系的，你要说有交叉吧，又不是我们FM，PNN，ONN等等网络中讲得这么回事。

这么看下来，DCN给我们的大体上是一个空头支票。它的交叉也不一定是我们想要的交叉。



----

参考：https://zhuanlan.zhihu.com/p/422368322