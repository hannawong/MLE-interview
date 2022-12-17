#### [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

请定义一个队列并实现函数 `max_value` 得到队列里的最大值，要求函数`max_value`、`push_back` 和 `pop_front` 的**均摊**时间复杂度都是O(1)。

若队列为空，`pop_front` 和 `max_value` 需要返回 -1

**示例 1：**

```
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
```

题解：

一个普通队列，用来记录真实的值；另外一个**单调队列**，具有单调栈的性质，但是可以pop队列头，所以算是双端队列。

![fig3.gif](https://pic.leetcode-cn.com/9d038fc9bca6db656f81853d49caccae358a5630589df304fc24d8999777df98-fig3.gif)



```python
class MaxQueue:

    def __init__(self):
        self.max_stack = []
        self.queue = []


    def max_value(self) -> int:
        if not self.max_stack: return -1
        return self.max_stack[0]


    def push_back(self, value: int) -> None:
        self.queue.append(value)
        while self.max_stack and self.max_stack[-1] < value: ##单调栈push
            self.max_stack.pop()
        self.max_stack.append(value)

    def pop_front(self) -> int:
        if not len(self.queue): return -1
        ans = self.queue[0]
        if ans == self.max_stack[0]: ##pop单调栈
            self.max_stack = self.max_stack[1:]
        self.queue = self.queue[1:]
        return ans
```


