#### [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

难度中等323

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

 

**示例 1：**

```
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

题解：栈混洗。

对于popped列表中的每个元素，要么直接去和pushed中对应的位置来匹配，要么从栈顶拿；否则，只能继续积累。

```python
class Solution:
    def validateStackSequences(self, pushed, popped) -> bool:
        pushed_idx = 0 ##指向pushed数组
        popped_idx = 0 ##指向popped数组

        stack = []
        while popped_idx < len(popped):
            if pushed_idx < len(popped) and popped[popped_idx] == pushed[pushed_idx]:##直接和push匹配上了
                pushed_idx+=1
                popped_idx+=1
            elif len(stack) and popped[popped_idx] == stack[-1]: ##和栈顶匹配上了
                popped_idx += 1
                stack.pop()
            else:##现在都匹配不上，sigh...，那么只能继续积累！
                if pushed_idx >= len(pushed):
                    return False
                stack.append(pushed[pushed_idx])
                pushed_idx+=1
        return len(stack) == 0
```

