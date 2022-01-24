# 向量

![img](https://pic3.zhimg.com/80/v2-03bb4b33e5a0f0f8da42ded343054d09_1440w.png)

#### 1. 动态空间管理

当数组的size要大于capacity时，需要开辟一段新的地址，容量加倍，然后把旧的元素都移过去：

![img](https://pic1.zhimg.com/80/v2-c5b0bde342f73c8ffb7e5f7fe65ab0b3_1440w.jpeg)

为什么要采用容量加倍而不是容量递增的做法呢？这是因为假如采用容量递增的方式，每次capacity不够就增加一个Increment，那么在初始容量为0的空向量中如果我连续插入 n = m*Increment个元素，在第1、1+Increment、1+2\*Increment、...1+(m-1)\*Increment次插入时都需要扩容。每次扩容即使不计入申请空间的操作，各次扩容复制原向量的时间成本依次为0, Increment, 2Increment, ...(m-1)\*Increment. 其中，m\*Increment = n, 那么总体耗时是O(n^2)的（等差数列求和）。每次扩容的分摊成本为O(n).

![img](https://pic1.zhimg.com/80/v2-22d86f866b658a30a8dc4b3b53d006f6_1440w.jpeg)

容量加倍的做法就不一样了。倘若在初始容量为1的向量中加入n = 2^m 个元素，那么在第1，2，4，8，...次插入时需要扩容。复制原向量的时间成本为1,2,4,8...2^m = n. 相加得到整体耗时为O(n)（等比数列求和）, 每次扩容分摊成本为O(1).  

![img](https://pic3.zhimg.com/80/v2-4a802f4d93ad5b31ef4a7b96a91be6c2_1440w.png)



#### 2. 基本操作

**元素访问：重载下标运算符[]**

如下语句的**返回值**是**引用**，因此可以用于**左值赋值**，例如：V[r] = 2*x+3

> C++ 函数可以返回一个引用，方式与返回一个指针类似。
>
> 当函数返回一个引用时，则返回一个指向返回值的隐式指针。这样，函数就可以放在赋值语句的左边。

```c++
template<typename T> T & Vector<T>::operator[](int r){return _elem[r];}
```

如下语句的返回值是**常量引用**，因此仅限于**右值**，不能赋值，如：int x = V[r]+V[s]

```c++
template<typename T> const T & Vector<T>::operator[](int r) const{return _elem[r];} 
```

这篇文章详细区分了函数返回值的值传递、引用传递、常量引用传递，需要仔细阅读！

https://blog.csdn.net/u012814856/article/details/84099328

**vector去重：**原地双指针法

```python
def deduplicate(nums):
    i = j = 0
    cnt = 0
    n = len(nums)
    while i < n and  j < n:
        if nums[i] == nums[j]:
            j += 1
        else: ##出现了不同元素！
            nums[cnt] = nums[i]
            cnt += 1
            i = j
    nums[cnt] = nums[i]
    print(nums)
```

**查找**： 

二分查找的平均查找长度计算：

![img](https://pica.zhimg.com/80/v2-39c6dc98c9247be4bc3737784785ad2a_1440w.jpeg)

每次target 比当前元素x小，查找长度+1；每次target 比当前元素大，或者命中，则查找长度+2.

平均成功查找长度、平均失败查找长度的计算如下。其中蓝色标为成功查找长度、橘色标为失败查找长度：

![img](https://pica.zhimg.com/80/v2-51004bbcccc4a81896a48fd770325296_1440w.jpeg)

一种常数项优化：斐波那契查找

![img](https://pica.zhimg.com/80/v2-98900764179b0008035ca7c1499ffcde_1440w.jpeg)

![img](https://pic2.zhimg.com/80/v2-04fc0d288e21d80d8808d0b8fdfa80d0_1440w.jpeg)

