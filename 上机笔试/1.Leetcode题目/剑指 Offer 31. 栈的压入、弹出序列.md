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
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        pushed_ptr = 0; popped_ptr = 0
        stack = []
        while pushed_ptr < len(pushed) or len(stack): ##【易错】两个条件，只要一个满足即可
            if len(stack) and stack[-1] == popped[popped_ptr]: ##和栈中匹配
                stack.pop()
                popped_ptr += 1
            else: ##和栈中不匹配，只能继续积累
                if pushed_ptr < len(pushed):
                    stack.append(pushed[pushed_ptr])
                    pushed_ptr += 1
                else: ##完全不行
                    return False
        return popped_ptr == len(popped)
                
```

