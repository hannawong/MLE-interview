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
        self.queue = []
        self.max_queue = []

    def max_value(self) -> int:
        if not len(self.queue):
            return -1
        return self.max_queue[0]

    def push_back(self, value: int) -> None:
        self.queue.append(value)
        if len(self.max_queue) == 0:
            self.max_queue.append(value)
        else:
            while len(self.max_queue):
                top = self.max_queue[-1]
                if value > top: ##维护单调栈性质
                    self.max_queue = self.max_queue[:-1]
                else:
                    break
            self.max_queue.append(value)

    def pop_front(self) -> int:
        if not len(self.queue):
            return -1
        if self.queue[0] == self.max_queue[0]:
            front = self.queue[0]
            self.queue = self.queue[1:]
            self.max_queue = self.max_queue[1:]
            return front
        else:
            front = self.queue[0]
            self.queue = self.queue[1:]
            return front
        



# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```

【总结】

单调栈：

```python
            while len(self.max_queue):
                top = self.max_queue[-1]
                if value > top: ##维护单调栈性质
                    self.max_queue = self.max_queue[:-1]
                else:
                    break
            self.max_queue.append(value)
```

