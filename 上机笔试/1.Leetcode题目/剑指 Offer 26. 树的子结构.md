#### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

难度中等537

输入两棵二叉树A和B，判断B是不是A的子结构。(**约定空树不是任意一个树的子结构)**

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

**示例 1：**

```
输入：A = [1,2,3], B = [3,1]
输出：false
```

**示例 2：**

```
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```



解法：

这类题是简单dfs+层次遍历所有的点。

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if B == None:
            return False
        
        def dfs(A,B):
            if A and not B:
                return True
            if not A and not B:
                return True
            if not A and B:
                return False
            if A.val!=B.val:
                return False
            return dfs(A.left,B.left) and dfs(A.right,B.right)
        
        queue = [A]
        ans = False
        while len(queue):
            front = queue[0]
            ans = ans or dfs(front,B)
            if front.left:
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
            queue = queue[1:]
        return ans
            
```

