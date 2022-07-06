#### [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

难度中等588

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

`     3    / \   4   5  / \ 1   2`
给定的树 B：

`   4   / 1`
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def is_sub(A,B): ##B是否是A的子结构
            if not A and B:
                return False
            if not B:
                return True
            if A.val != B.val:
                return False
            return is_sub(A.left,B.left) and is_sub(A.right,B.right)
        
        def dfs(A,B):
            if not A and B:
                return False
            if not B:
                return True
            if A.val == B.val:
                return is_sub(A,B) or dfs(A.left,B) or dfs(A.right,B)
            else:
                return dfs(A.left,B) or dfs(A.right,B)
        if B == None:
            return False
        return dfs(A,B)
```

