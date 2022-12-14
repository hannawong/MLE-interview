#### [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

难度中等

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

`     3    / \   4   5  / \ 1   2`
给定的树 B：

`   4   / 1`
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。



方法一：利用层次遍历，每次都判断B是不是当前节点的子树

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def helper(A,B):
            if B == None: return True
            if A == None and B != None: return False
            if B.val != A.val: return False
            return helper(A.left,B.left) and helper(A.right,B.right)
        if B == None: return False
        queue = deque([A])
        while len(queue):
            front = queue[0]
            if helper(front,B): return True
            if front.left:
                queue.append(front.left)
            if front.right:
                queue.append(front.right)
            queue.popleft()
        return False
```

方法二：DFS来替换层次遍历也可

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def helper(A,B):
            if B == None: return True
            if A == None and B != None: return False
            if B.val != A.val: return False
            return helper(A.left,B.left) and helper(A.right,B.right)
        def dfs(A,B):
            if B == None: return True
            if A == None and B != None: return False
            if A.val == B.val and helper(A,B): return True
            return dfs(A.left,B) or dfs(A.right,B)
        if B == None: return False
        return dfs(A,B)
```

