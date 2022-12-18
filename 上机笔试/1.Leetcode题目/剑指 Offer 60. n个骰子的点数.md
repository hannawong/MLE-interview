#### [剑指 Offer 60. n个骰子的点数](https://leetcode.cn/problems/nge-tou-zi-de-dian-shu-lcof/)

难度中等419

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

 

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

 

**示例 1:**

```
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
```



题解：

用前一轮的dfs生成的list来继续dfs。

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        if n == 1: return [1/6] * 6
        prev_prob = self.dicesProbability(n-1)
        new_prob = [0] * (len(prev_prob)+6)
        for i in range(len(prev_prob)):
            for j in range(1,7):
                new_prob[i+j] += prev_prob[i] / 6
        return new_prob[1:]

```

