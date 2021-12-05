# 线性回归, L1/L2正则

### 1. 线性回归

基本假设：误差是均值为0的正态分布

![img](https://pic1.zhimg.com/80/v2-6d056385519566a4722c02f86cbebabb_1440w.png)

目标就是拟合 ![W,b](https://www.zhihu.com/equation?tex=W%2Cb)  ；在损失函数为MSE时，此问题就是最小二乘问题，有close-form solution:

​                                                          ![W = (X^TX)^{-1}X^Ty](https://www.zhihu.com/equation?tex=W%20%3D%20(X%5ETX)%5E%7B-1%7DX%5ETy)  

### 2. Bias-Variance Tradeoff

![img](https://pica.zhimg.com/80/v2-35d08dc52ec8130ecfd53ff90f8a07fa_1440w.png)

### 2.1 L2正则化 --- ridge regression

增加了参数平方和<c的限制：

![img](https://pica.zhimg.com/80/v2-516c9054cd869303df5e14bf49dbd236_1440w.png)

损失函数增加L2正则项：

![img](https://pic2.zhimg.com/80/v2-e52976e32aa66a62184e1031386d297a_1440w.png)

其实，ridge regression也是有close-form solution的：

​                                                           ![W = (X^TX+\alpha I)^{-1}X^Ty](https://www.zhihu.com/equation?tex=W%20%3D%20(X%5ETX%2B%5Calpha%20I)%5E%7B-1%7DX%5ETy)  


  



