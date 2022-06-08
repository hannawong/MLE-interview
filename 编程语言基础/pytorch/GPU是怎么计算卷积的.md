# GPU是怎么计算卷积的

![img](https://pic3.zhimg.com/80/v2-a44270779bf66d576794d5e3e698c057_1440w.png)

![img](https://pic3.zhimg.com/80/v2-86d70fd9d7cbe71d5f6dd6dce682b5ff_1440w.png)

![img](https://pica.zhimg.com/80/v2-57a9d68859c8520a32b7ec58859a2787_1440w.png)

![img](https://pic1.zhimg.com/80/v2-9f8ea93e10b60003456f2c3d6885db5f_1440w.png)

![img](https://pica.zhimg.com/80/v2-07718a326e8a872440cbad0f3375cddf_1440w.png)

注意，由于并行训练，复杂度是log级别的。这种方法的好处比起暴力解法而言，就是可以并行计算。

![img](https://pic1.zhimg.com/80/v2-223af744c9021e4a8ff01dff41c1ff88_1440w.png)