#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode.cn/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

难度简单326

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

 

**示例:**

```
输入: a = 1, b = 1
输出: 2
```

 

**提示：**

- `a`, `b` 均可能是负数或 0
- 结果不会溢出 32 位整数



题解：

通过与运算与异或运算来实现加法运算 

- 计算两个数不算进位的结果 (a^b) 
- 计算两个数进位的结果 (a&b)<<1 
- 将两个结果相加.我们发现又要用到加法运算,那么其实我们重复上述步骤就行了,直到一个数变 为0(不再进位)运算全部完成

```c++
class Solution {
public:
    int add(int a, int b) {
        if(a == 0) return b;
        if (b == 0) return a;
        int wo_carry = a^b;
        int carry = (unsigned int)(a&b) << 1;
        return add(wo_carry,carry);
    }
};
```

