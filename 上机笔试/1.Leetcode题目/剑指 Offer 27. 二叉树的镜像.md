#### [剑指 Offer 27. 二叉树的镜像](https://leetcode.cn/problems/er-cha-shu-de-jing-xiang-lcof/)

难度简单267

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

**示例 1：**

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

递归法自然简单，也需掌握迭代法：层序遍历，每次都交换左右孩子的位置

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        queue = [root]
        while len(queue):
            front = queue[0]
            queue = queue[1:]
            left, right  =None,None
            if front.left:
                left = front.left
                queue.append(front.left)
            if front.right:
                right = front.right
                queue.append(front.right)
            front.left = right
            front.right = left
        return root
```

