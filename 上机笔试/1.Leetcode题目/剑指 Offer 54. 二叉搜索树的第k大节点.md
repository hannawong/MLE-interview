#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

难度简单

给定一棵二叉搜索树，请找出其中第 `k` 大的节点的值。

 

**示例 1:**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```



题解：反中序遍历 + 剪枝

```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        self.cnt = 0
        self.ans = -1
        def inorder(root):
            if not root:
                return 
            inorder(root.right)
            #print(root.val)
            self.cnt += 1
            if self.cnt == k:
                self.ans = root.val
                return 
            inorder(root.left)
        inorder(root)
        return self.ans
```



时间复杂度为O(树高+k)，因为在开始遍历之前我们还需要到达**最底下的全局第一个访问位置**。所以，平均情况下，复杂度为O(logn+k)；最坏情况下复杂度为O(n+k). 