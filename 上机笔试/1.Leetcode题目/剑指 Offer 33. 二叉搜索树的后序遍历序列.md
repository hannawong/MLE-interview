#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

难度中等482

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：

```
     5
    / \
   2   6
  / \
 1   3
```

**示例 1：**

```
输入: [1,6,3,2,5]
输出: false
```



答案：

```python
class Solution:
    def verifyPostorder(self, postorder) -> bool:
        if len(postorder) <= 1: ##单个节点，或者None，必定为二叉搜索树
            return True
        root = postorder[-1]
        left_postorder = []
        right_postorder = []
        for i in range(0,len(postorder)-1): ##从前往后看
            if postorder[i] < root: ##还在遍历左子树
                if len(right_postorder) >= 1: ##什么！居然右子树都不为空了！这一定不对
                    return False
                else:
                    left_postorder.append(postorder[i]) 
            else:
                right_postorder.append(postorder[i])##开始遍历右子树
        return self.verifyPostorder(left_postorder) and self.verifyPostorder(right_postorder)
```

