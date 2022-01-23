# 绪论

### 1. 复杂度

大O记号：如果存在c > 0, 使得当n趋于正无穷时， T(n) < c f(n), 则 T(n) = O(f(n)). 简记：大O记号括号中的函数要渐进地更大一些。

大 ![\Omega](https://www.zhihu.com/equation?tex=%5COmega)记号：恰和大O记号相反，如果存在 c > 0, 使得当n趋于正无穷时，T(n) > c f(n), 则 ![T(n) = \Omega(f(n))](https://www.zhihu.com/equation?tex=T(n)%20%3D%20%5COmega(f(n)))  . 简记：大Omega记号括号中的函数要渐进地更小一些。

大 ![\Theta](https://www.zhihu.com/equation?tex=%5CTheta)  记号：“夹逼”。如果存在 ![c_1 > c_2 > 0 ](https://www.zhihu.com/equation?tex=c_1%20%3E%20c_2%20%3E%200%20)  使得 ![c_1 f(n) > T(n) > c_2 f(n)](https://www.zhihu.com/equation?tex=c_1%20f(n)%20%3E%20T(n)%20%3E%20c_2%20f(n))，则 ![T(n) = \Theta(f(n))](https://www.zhihu.com/equation?tex=T(n)%20%3D%20%5CTheta(f(n)))  

![img](https://pica.zhimg.com/80/v2-b8052486fff43c70f7f468494b5937e2_1440w.jpeg)



Master Theorem:

![img](https://pic1.zhimg.com/80/v2-ca7830ee5d665aa8d4437322830bac60_1440w.jpg)

